/**
 * VR Controller for WebXR integration
 * Handles Vision Pro VR mode
 */

export class VRController {
    constructor(renderer, volumeRenderer = null) {
        this.renderer = renderer;
        this.volumeRenderer = volumeRenderer;
        this.vrButton = null;
        this.spatialButton = null;
        this.isVRSupported = false;
        this.isInVR = false;
        this.controllers = [];
        this.availableModes = { ar: false, vr: false };
        this.preferredSessionMode = null;
        this.activeSessionMode = null;
        this.forceARPassthrough = this.detectVisionPro();
        this.isSecureContext = window.isSecureContext;
        this.xrElements = {
            secure: document.getElementById('xr-secure-status'),
            spatial: document.getElementById('xr-spatial-status'),
            blend: document.getElementById('xr-blend-status'),
            warning: document.getElementById('xr-warning')
        };
        this.currentWarning = '';
        this.currentWarningSeverity = 'info';
        this.detectedBlendMode = null;
        this.blendWarningShown = false;

        if (!this.isSecureContext) {
            this.setXRWarning('Vision ProのパススルーにはHTTPS接続が必要です。', 'error');
        }

        this.init();
    }

    async init() {
        // Check if WebXR is supported
        console.log('Checking WebXR support...');
        console.log('navigator.xr exists:', 'xr' in navigator);
        console.log('Window is secure context:', window.isSecureContext);
        console.log('Current URL:', window.location.href);

        if ('xr' in navigator) {
            try {
                // Check for both VR and AR support (Vision Pro uses AR mode)
                const vrSupported = await navigator.xr.isSessionSupported('immersive-vr');
                const arSupported = await navigator.xr.isSessionSupported('immersive-ar');

                this.availableModes.vr = vrSupported;
                this.availableModes.ar = arSupported;
                this.isVRSupported = this.availableModes.vr || this.availableModes.ar;
                this.preferredSessionMode = this.choosePreferredSessionMode();

                console.log('VR support:', vrSupported);
                console.log('AR support:', arSupported);
                console.log('Session mode preference:', this.preferredSessionMode);
                if (this.forceARPassthrough) {
                    console.log('Vision Pro detected -> forcing passthrough (immersive-ar) when possible');
                    if (!arSupported) {
                        this.setXRWarning('Vision Proを検出しましたがSafariがimmersive-arを許可していません。Feature Flagsの"WebXR Device API"とHTTPSを確認してください。', 'warn');
                    }
                }

                if (this.isVRSupported) {
                    this.setupVRButton();
                } else {
                    console.log('Neither immersive VR nor AR is supported on this device');
                    console.log('This might be because:');
                    console.log('1. WebXR is not enabled in Safari settings');
                    console.log('2. You are not using HTTPS (required for IP addresses)');
                    console.log('3. Your device does not support WebXR');
                    this.showVRNotSupported();
                }
            } catch (error) {
                console.error('Error checking VR/AR support:', error);
                this.showVRNotSupported();
            }
        } else {
            console.log('WebXR is not available');
            console.log('For Vision Pro Safari:');
            console.log('1. Go to Settings > Safari > Advanced > Feature Flags');
            console.log('2. Enable "WebXR Device API"');
            console.log('3. Reload this page');
            this.showVRNotSupported('WebXR not available - Check Safari settings');
        }

        this.updateXRDiagnostics();
    }

    setupVRButton() {
        this.vrButton = document.getElementById('vr-button');
        this.spatialButton = document.getElementById('spatial-button');

        if (!this.vrButton) return;

        this.vrButton.disabled = false;
        this.updateAllButtonLabels();

        // Custom XR session handler
        this.vrButton.addEventListener('click', async () => {
            if (this.isInVR) {
                // Exit XR session
                const session = this.renderer.xr.getSession();
                if (session) {
                    await session.end();
                }
                return;
            }

            // Enter XR session
            try {
                await this.startPreferredXRSession();
            } catch (error) {
                console.error('Failed to start XR session:', error);
                alert(`Failed to start XR session: ${error.message}`);
            }
        });

        if (this.spatialButton) {
            this.spatialButton.style.display = 'block';
            this.spatialButton.disabled = true;
            this.updateSpatialButtonLabel(this.spatialButton);

            this.spatialButton.addEventListener('click', async () => {
                const spatialStatus = this.describeSpatialCapability();
                if (!spatialStatus.available) {
                    alert(spatialStatus.message);
                    this.setXRWarning(spatialStatus.message, 'error');
                    return;
                }

                if (this.isInVR && this.activeSessionMode === 'immersive-ar') {
                    const session = this.renderer.xr.getSession();
                    if (session) {
                        await session.end();
                    }
                    return;
                }

                try {
                    await this.startXRSessionForMode('immersive-ar');
                    this.preferredSessionMode = 'immersive-ar';
                } catch (error) {
                    console.error('Failed to start AR passthrough session:', error);
                    alert(`Failed to start Spatial Mode: ${error.message}`);
                    this.setXRWarning(`Spatial Mode start failed: ${error.message}`, 'error');
                }
            });
        }

        // Listen for VR/AR session events
        this.renderer.xr.addEventListener('sessionstart', async () => {
            console.log('XR session started');
            const session = this.renderer.xr.getSession();
            if (session) {
                await this.configureXRRenderLayer(session, this.activeSessionMode === 'immersive-ar');
                this.detectedBlendMode = session.environmentBlendMode || null;
                if (this.activeSessionMode === 'immersive-ar' && session.environmentBlendMode !== 'alpha-blend') {
                    this.handlePassthroughBlocked(session.environmentBlendMode);
                }
            }
            this.isInVR = true;
            this.updateAllButtonLabels();
            this.updateXRDiagnostics();
            this.onSessionStart();
        });

        this.renderer.xr.addEventListener('sessionend', () => {
            console.log('XR session ended');
            this.isInVR = false;
            this.activeSessionMode = null;
            this.updateAllButtonLabels();
            this.onSessionEnd();
            this.detectedBlendMode = null;
            this.updateXRDiagnostics();
        });

        console.log('XR buttons configured');
        this.updateXRDiagnostics();
    }

    showVRNotSupported(message = 'VR Not Supported') {
        const customButton = document.getElementById('vr-button');
        const spatialButton = document.getElementById('spatial-button');
        if (customButton) {
            customButton.disabled = true;
            customButton.textContent = message;
            customButton.style.fontSize = '12px'; // Smaller font for longer messages
        }
        if (spatialButton) {
            spatialButton.disabled = true;
            spatialButton.textContent = 'Spatial Mode unavailable in this browser';
        }
        this.setXRWarning(message, 'error');
        this.updateXRDiagnostics();
    }

    updateVRButtonLabel(button) {
        if (!button) return;

        const mode = this.isInVR ? this.activeSessionMode : this.preferredSessionMode;
        const isAR = mode === 'immersive-ar';

        if (this.isInVR) {
            button.textContent = isAR ? 'Exit Spatial Mode' : 'Exit VR Mode';
        } else {
            if (isAR) {
                button.textContent = 'Enter Spatial Mode (Passthrough)';
            } else {
                button.textContent = 'Enter VR Mode';
            }
        }
    }

    updateSpatialButtonLabel(button) {
        if (!button) return;

        const spatialStatus = this.describeSpatialCapability();
        button.style.display = 'block';

        if (!spatialStatus.available) {
            button.textContent = spatialStatus.message;
            button.disabled = true;
            return;
        }

        if (this.isInVR && this.activeSessionMode === 'immersive-ar') {
            button.textContent = 'Exit Spatial Mode';
            button.disabled = false;
        } else if (this.isInVR && this.activeSessionMode !== 'immersive-ar') {
            button.textContent = 'Spatial Mode Unavailable (Exit VR First)';
            button.disabled = true;
        } else {
            button.textContent = 'Enter Spatial Mode (Passthrough)';
            button.disabled = false;
        }
    }

    updateAllButtonLabels() {
        this.updateVRButtonLabel(this.vrButton);
        this.updateSpatialButtonLabel(this.spatialButton);
    }

    describeSpatialCapability() {
        if (!this.isSecureContext) {
            return {
                available: false,
                message: 'Spatial Mode requires HTTPS (Secure Context)'
            };
        }

        if (!this.availableModes.ar) {
            if (this.forceARPassthrough) {
                return {
                    available: false,
                    message: 'Vision Pro detected - enable WebXR Device API in Safari settings'
                };
            }
            return {
                available: false,
                message: 'Spatial Mode unavailable (immersive-ar not supported)'
            };
        }

        return { available: true };
    }

    hasSpatialSupport() {
        return this.describeSpatialCapability().available;
    }

    setXRWarning(message = '', severity = 'info') {
        this.currentWarning = message || '';
        this.currentWarningSeverity = message ? severity : 'info';
        if (this.xrElements.warning) {
            const text = this.currentWarning || '---';
            this.xrElements.warning.textContent = text;
            this.xrElements.warning.setAttribute('data-severity', this.currentWarning ? severity : 'info');
        }
    }

    applyStatusClass(element, state) {
        if (!element) return;
        element.classList.remove('ok', 'warn', 'error');
        if (state) {
            element.classList.add(state);
        }
    }

    updateXRDiagnostics() {
        this.isSecureContext = window.isSecureContext;

        if (this.xrElements.secure) {
            this.xrElements.secure.textContent = this.isSecureContext ? 'HTTPS / Secure' : 'HTTP / Blocked';
            this.applyStatusClass(this.xrElements.secure, this.isSecureContext ? 'ok' : 'error');
        }

        if (this.xrElements.spatial) {
            const spatialStatus = this.describeSpatialCapability();
            if (spatialStatus.available) {
                this.xrElements.spatial.textContent = 'immersive-ar ready';
                this.applyStatusClass(this.xrElements.spatial, 'ok');
            } else {
                this.xrElements.spatial.textContent = spatialStatus.message;
                const state = this.isSecureContext ? 'warn' : 'error';
                this.applyStatusClass(this.xrElements.spatial, state);
            }
        }

        if (this.xrElements.blend) {
            const blendMode = this.detectedBlendMode || '---';
            this.xrElements.blend.textContent = blendMode;
            const blendState = blendMode === 'alpha-blend' ? 'ok' : (blendMode === '---' ? null : 'warn');
            this.applyStatusClass(this.xrElements.blend, blendState);
        }

        if (this.xrElements.warning) {
            const warningText = this.currentWarning || '---';
            this.xrElements.warning.textContent = warningText;
            this.xrElements.warning.setAttribute('data-severity', this.currentWarning ? this.currentWarningSeverity : 'info');
        }
    }

    handlePassthroughBlocked(blendMode) {
        const reason = this.isSecureContext
            ? `Vision ProがenvironmentBlendMode="${blendMode}"を返しました。HTTPS接続とSafariのWebXR設定を再確認してください。`
            : 'Vision ProはHTTP接続でパススルーを許可しません。HTTPSでアクセスしてください。';
        this.setXRWarning(reason, 'error');
        if (!this.blendWarningShown) {
            alert(reason);
            this.blendWarningShown = true;
        }
    }

    detectVisionPro() {
        const ua = navigator.userAgent || '';
        const isVisionPro = /VisionOS|Vision Pro|AppleVision/i.test(ua);
        if (isVisionPro) {
            console.log('Vision Pro user agent detected:', ua);
        }
        return isVisionPro;
    }

    choosePreferredSessionMode() {
        if (this.availableModes.ar) {
            return 'immersive-ar';
        }
        if (this.availableModes.vr) {
            return 'immersive-vr';
        }
        return null;
    }

    buildModeCandidates() {
        const modes = [];
        if (this.preferredSessionMode) {
            modes.push(this.preferredSessionMode);
        }
        if (this.availableModes.ar && !modes.includes('immersive-ar')) {
            modes.push('immersive-ar');
        }
        if (this.availableModes.vr && !modes.includes('immersive-vr')) {
            modes.push('immersive-vr');
        }
        return modes;
    }

    async startPreferredXRSession() {
        const modes = this.buildModeCandidates();
        if (!modes.length) {
            throw new Error('No XR modes are available');
        }

        let lastError = null;
        for (const mode of modes) {
            try {
                await this.startXRSessionForMode(mode);
                this.preferredSessionMode = mode;
                return;
            } catch (error) {
                console.warn(`Failed to start ${mode}:`, error);
                lastError = error;
            }
        }

        if (lastError) {
            throw lastError;
        }
    }

    async startXRSessionForMode(mode) {
        const isARSession = mode === 'immersive-ar';
        const referenceSpace = isARSession ? 'local' : 'local-floor';
        this.renderer.xr.setReferenceSpaceType(referenceSpace);

        const sessionInit = isARSession ?
            {
                requiredFeatures: ['local'],
                optionalFeatures: ['local-floor', 'dom-overlay', 'layers', 'plane-detection'],
                domOverlay: { root: document.body }
            } :
            {
                requiredFeatures: ['local-floor'],
                optionalFeatures: ['layers'],
            };

        const session = await navigator.xr.requestSession(mode, sessionInit);
        await this.configureXRRenderLayer(session, isARSession);
        await this.renderer.xr.setSession(session);
        this.activeSessionMode = mode;
        console.log('XR session started successfully');
        console.log('Session mode:', mode);
        console.log('Environment blend mode:', session.environmentBlendMode);
    }

    onSessionStart() {
        // Hide UI elements that shouldn't be visible in VR/AR
        const controls = document.getElementById('controls');
        const infoOverlay = document.getElementById('info-overlay');

        if (controls) controls.style.display = 'none';
        if (infoOverlay) infoOverlay.style.display = 'none';

        console.log('=== XR Session Start Debug ===');
        console.log('Session mode:', this.activeSessionMode);

        // Configure rendering based on session mode
        if (this.volumeRenderer) {
            this.volumeRenderer.enterXRMode(this.activeSessionMode || this.preferredSessionMode);
        }

        // Get session to check environment blend mode
        const session = this.renderer.xr.getSession();
        if (session) {
            console.log('Environment blend mode:', session.environmentBlendMode);
            console.log('Expected: "alpha-blend" for AR passthrough, "opaque" for VR');
            if (session.environmentBlendMode !== 'alpha-blend' && this.activeSessionMode === 'immersive-ar') {
                console.warn('Passthrough not active despite AR mode. Check HTTPS and feature flags.');
            }
        }

        // Debug renderer settings
        console.log('Canvas has alpha:', this.renderer.getContext().getContextAttributes().alpha);
        console.log('Renderer autoclear:', this.renderer.autoClear);

        // Setup VR/AR controllers
        this.setupVRControllers();

        console.log('=== Entering XR mode ===');
    }

    async configureXRRenderLayer(session, isARSession) {
        if (!session || !isARSession) {
            return;
        }

        if (typeof XRWebGLLayer === 'undefined') {
            console.warn('XRWebGLLayer is not available in this browser');
            return;
        }

        try {
            const gl = this.renderer.getContext();
            if (gl.makeXRCompatible) {
                await gl.makeXRCompatible();
            }

            const layerInit = {
                alpha: true,
                antialias: true,
                depth: true,
                stencil: false,
                framebufferScaleFactor: this.renderer.getPixelRatio()
            };

            session.updateRenderState({
                baseLayer: new XRWebGLLayer(session, gl, layerInit)
            });

            console.log('Configured transparent XRWebGLLayer for AR passthrough');
        } catch (layerError) {
            console.warn('Failed to configure custom XRWebGLLayer:', layerError);
        }
    }

    setupVRControllers() {
        // Get VR controllers
        const controller0 = this.renderer.xr.getController(0);
        const controller1 = this.renderer.xr.getController(1);

        // Add event listeners for controller buttons
        controller0.addEventListener('selectstart', () => this.onSelectStart(0));
        controller0.addEventListener('selectend', () => this.onSelectEnd(0));
        controller1.addEventListener('selectstart', () => this.onSelectStart(1));
        controller1.addEventListener('selectend', () => this.onSelectEnd(1));

        // Store controllers
        this.controllers = [controller0, controller1];

        console.log('VR controllers configured');
    }

    onSelectStart(controllerIndex) {
        console.log(`Controller ${controllerIndex} select start`);

        // When trigger is pressed, increment slice position
        if (this.volumeRenderer) {
            // Controller 0 (left hand): adjust axial slice
            // Controller 1 (right hand): adjust sagittal slice
            if (controllerIndex === 0) {
                const currentPos = this.volumeRenderer.axialPosition;
                const newPos = Math.min(1.0, currentPos + 0.05);
                this.volumeRenderer.setAxialPosition(newPos);
            } else {
                const currentPos = this.volumeRenderer.sagittalPosition;
                const newPos = Math.min(1.0, currentPos + 0.05);
                this.volumeRenderer.setSagittalPosition(newPos);
            }
        }
    }

    onSelectEnd(controllerIndex) {
        console.log(`Controller ${controllerIndex} select end`);
    }

    async onSessionEnd() {
        console.log('=== XR Session End ===');

        // Restore UI elements
        const controls = document.getElementById('controls');
        const infoOverlay = document.getElementById('info-overlay');

        if (controls) controls.style.display = 'flex';
        if (infoOverlay) infoOverlay.style.display = 'block';

        // Restore scene background
        if (this.volumeRenderer) {
            this.volumeRenderer.exitXRMode();
        }

        console.log('Exiting XR mode - UI restored');
    }

    /**
     * Check if currently in VR mode
     */
    isInVRMode() {
        return this.isInVR;
    }

    /**
     * Check if VR is supported
     */
    isSupported() {
        return this.isVRSupported;
    }

    /**
     * Set the volume renderer for VR controller interaction
     */
    setVolumeRenderer(volumeRenderer) {
        this.volumeRenderer = volumeRenderer;
        console.log('VolumeRenderer set for VR controller');
    }
}
