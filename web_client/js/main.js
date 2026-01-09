/**
 * Main application entry point
 * Coordinates all components
 */

import { apiClient } from './api_client.js?v=12';
import { VolumeRenderer } from './volume_renderer_v8.js?v=12';
import { VRController } from './vr_controller.js?v=12';

class ShioRIS3Viewer {
    constructor() {
        this.ensureDiagnosticsPanel();
        this.apiClient = apiClient;
        this.volumeRenderer = null;
        this.vrController = null;
        this.currentPatientId = null;
        this.buildInfo = null;
        this.localStorageVersionKey = 'shioris3:webVersion';
        this.diagnosticsRoot = document.getElementById('diagnostics-inline');
        this.buildVersionElement = document.getElementById('build-version-text');
        this.buildStatusElement = document.getElementById('build-version-status');
        this.buildEndpointElement = document.getElementById('build-version-endpoint');
        this.secureContextElement = document.getElementById('secure-context-status');
        this.cacheBypassNoteElement = document.getElementById('cache-bypass-note');
        this.cacheBypassButton = document.getElementById('cache-bypass-btn');

        // Debounce timers for slice position updates
        this.axialDebounceTimer = null;
        this.sagittalDebounceTimer = null;
        this.coronalDebounceTimer = null;
        this.sliceDebounceDelay = 250; // ms

        this.updateEnvironmentDiagnostics();
        this.init();
    }

    ensureDiagnosticsPanel() {
        let diagnostics = document.getElementById('diagnostics-inline');
        if (!diagnostics) {
            const headerRight = document.getElementById('header-right');
            diagnostics = document.createElement('div');
            diagnostics.id = 'diagnostics-inline';
            diagnostics.innerHTML = this.getDiagnosticsTemplate();

            if (headerRight) {
                headerRight.appendChild(diagnostics);
            } else {
                diagnostics.dataset.fallback = 'true';
                document.body.appendChild(diagnostics);
            }
        } else {
            const versionLine = document.getElementById('build-version-text')?.parentElement || null;
            this.ensureDiagnosticsField(diagnostics, 'build-version-endpoint', 'Origin', 'URLを確認中...', versionLine);
            this.ensureDiagnosticsField(diagnostics, 'secure-context-status', 'Security', '---', versionLine);
            this.ensureDiagnosticsField(diagnostics, 'build-version-status', 'Status', '---');
            this.ensureCacheBypassLine(diagnostics);
        }
    }

    getDiagnosticsTemplate() {
        return `
            <div class="diagnostics-line">
                <span class="label">Origin</span>
                <span id="build-version-endpoint">URLを確認中...</span>
            </div>
            <div class="diagnostics-line">
                <span class="label">Security</span>
                <span id="secure-context-status">---</span>
            </div>
            <div class="diagnostics-line">
                <span class="label">Version</span>
                <span id="build-version-text">Webビルド情報を取得中...</span>
            </div>
            <div class="diagnostics-line">
                <span class="label">Status</span>
                <span id="build-version-status">---</span>
            </div>
            <div class="diagnostics-line">
                <button id="cache-bypass-btn" disabled>キャッシュ無効化リロード</button>
                <span id="cache-bypass-note">Vision Proでバージョンが一致しない場合はキャッシュ削除リロードを実行してください。<strong>HTTPS未設定のままではVision Proで没入表示できません。</strong></span>
            </div>
        `;
    }

    ensureDiagnosticsField(root, id, label, defaultText, beforeElement = null) {
        if (document.getElementById(id)) {
            return;
        }

        const line = document.createElement('div');
        line.className = 'diagnostics-line';

        const labelElement = document.createElement('span');
        labelElement.className = 'label';
        labelElement.textContent = label;

        const valueElement = document.createElement('span');
        valueElement.id = id;
        valueElement.textContent = defaultText;

        line.appendChild(labelElement);
        line.appendChild(valueElement);

        if (beforeElement && beforeElement.parentElement === root) {
            root.insertBefore(line, beforeElement);
        } else {
            root.appendChild(line);
        }
    }

    ensureCacheBypassLine(root) {
        const button = document.getElementById('cache-bypass-btn');
        const note = document.getElementById('cache-bypass-note');
        if (button && note) {
            return;
        }

        const line = document.createElement('div');
        line.className = 'diagnostics-line';

        const originalButtonParent = button?.parentElement || null;
        const originalNoteParent = note?.parentElement || null;

        const reloadButton = button || document.createElement('button');
        reloadButton.id = 'cache-bypass-btn';
        reloadButton.disabled = true;
        reloadButton.textContent = 'キャッシュ無効化リロード';

        const noteSpan = note || document.createElement('span');
        noteSpan.id = 'cache-bypass-note';
        noteSpan.innerHTML = 'Vision Proでバージョンが一致しない場合はキャッシュ削除リロードを実行してください。<strong>HTTPS未設定のままではVision Proで没入表示できません。</strong>';

        line.appendChild(reloadButton);
        line.appendChild(noteSpan);
        root.appendChild(line);

        const maybeRemoveLine = (elementParent) => {
            if (elementParent && elementParent !== line && elementParent.classList.contains('diagnostics-line') && elementParent.childElementCount === 0) {
                elementParent.remove();
            }
        };

        maybeRemoveLine(originalButtonParent);
        maybeRemoveLine(originalNoteParent);
    }

    updateEnvironmentDiagnostics() {
        if (this.buildEndpointElement) {
            const url = window.location.href;
            const protocol = window.location.protocol;
            const host = window.location.host;
            this.buildEndpointElement.textContent = `${protocol}//${host}`;
            this.buildEndpointElement.setAttribute('title', url);
        }

        if (this.secureContextElement) {
            const isSecure = window.isSecureContext;
            this.secureContextElement.textContent = isSecure ? 'Secure (HTTPS)' : 'Not secure (HTTP)';
            this.secureContextElement.classList.toggle('secure', isSecure);
            this.secureContextElement.classList.toggle('insecure', !isSecure);
        }

        if (this.cacheBypassNoteElement && !window.isSecureContext) {
            this.cacheBypassNoteElement.innerHTML = 'Vision Proでバージョンが一致しない場合はキャッシュ削除リロードを実行してください。<strong>HTTPS未設定のままではVision Proで没入表示できません。</strong>';
        }
    }

    async init() {
        console.log('Initializing ShioRIS3 Viewer...');
        this.setupCacheBypassButton();
        await this.loadBuildInfo();

        try {
            // Initialize 3D renderer
            const canvas = document.getElementById('viewer-canvas');
            if (!canvas) {
                throw new Error('Canvas element not found');
            }
            console.log('Canvas found:', canvas);

            this.volumeRenderer = new VolumeRenderer(canvas);
            console.log('VolumeRenderer created');

            // Initialize VR controller
            this.vrController = new VRController(this.volumeRenderer.getRenderer(), this.volumeRenderer);
            console.log('VRController created');

            // Test connection to backend
            await this.testConnection();

            // Load patient list
            await this.loadPatients();

            // Setup UI event listeners
            this.setupEventListeners();

            // Hide loading indicator
            this.hideLoading();

            console.log('ShioRIS3 Viewer initialized successfully');
        } catch (error) {
            console.error('Failed to initialize ShioRIS3 Viewer:', error);
            this.hideLoading();
            alert(`Initialization failed: ${error.message}\n\nPlease check the browser console for details.`);
        }
    }

    async testConnection() {
        const statusElement = document.getElementById('connection-status');

        try {
            const success = await this.apiClient.testConnection();
            if (success) {
                statusElement.textContent = 'Connected to ShioRIS3';
                statusElement.classList.add('connected');
                console.log('Successfully connected to backend');
            } else {
                throw new Error('Connection test failed');
            }
        } catch (error) {
            statusElement.textContent = 'Connection Failed';
            statusElement.classList.add('error');
            console.error('Failed to connect to backend:', error);
        }
    }

    async loadPatients() {
        try {
            const data = await this.apiClient.getPatients();
            console.log('Loaded patients:', data);

            const select = document.getElementById('patient-select');
            select.innerHTML = '<option value="">Select a patient...</option>';

            if (data.patients && data.patients.length > 0) {
                data.patients.forEach(patient => {
                    const option = document.createElement('option');
                    option.value = patient.id;
                    option.textContent = `${patient.name} (${patient.studyDate})`;
                    select.appendChild(option);
                });
            } else {
                select.innerHTML = '<option value="">No patients available</option>';
            }
        } catch (error) {
            console.error('Failed to load patients:', error);
            const select = document.getElementById('patient-select');
            select.innerHTML = '<option value="">Error loading patients</option>';
        }
    }

    async loadPatient(patientId) {
        if (!patientId) {
            console.warn('No patient ID provided');
            return;
        }

        this.currentPatientId = patientId;
        console.log('Loading patient:', patientId);

        this.showLoading('Loading volume metadata...');

        try {
            // Load volume metadata
            const volumeData = await this.apiClient.getVolume(patientId);
            console.log('Volume data:', volumeData);

            if (volumeData.status === 'error') {
                throw new Error(volumeData.message);
            }

            // Create volume visualization with real DICOM slices
            this.showLoading('Loading DICOM slices...');
            await this.volumeRenderer.createVolumeFromData(volumeData, this.apiClient, patientId);

            // Try to load structures (will fail silently if not available)
            // Use simplify=3 (every 3rd point) to reduce data size for VR
            if (document.getElementById('show-structures').checked) {
                try {
                    this.showLoading('Loading RT structures...');
                    const structureData = await this.apiClient.getStructures(patientId, 3);
                    if (structureData.status === 'success') {
                        this.volumeRenderer.addStructures(structureData);
                        console.log(`RT Structures loaded: ${structureData.totalPoints} points (${structureData.simplification}x simplification, ${structureData.message})`);
                    }
                } catch (error) {
                    console.log('RT Structures not available:', error.message);
                }
            }

            // Try to load dose (will fail silently if not available)
            if (document.getElementById('show-dose').checked) {
                try {
                    this.showLoading('Loading dose isosurfaces...');
                    const doseData = await this.apiClient.getDose(patientId);
                    if (doseData.status === 'success') {
                        if (doseData.isosurfaceCount > 0) {
                            this.volumeRenderer.addDose(doseData);
                            console.log(`Dose data loaded: ${doseData.isosurfaceCount} isosurfaces, ${doseData.totalTriangles} triangles`);
                        } else {
                            console.log('No dose isosurfaces available. Generate them in ShioRIS3 using "3D Isosurface" button.');
                        }
                    }
                } catch (error) {
                    console.warn('Failed to load dose data:', error.message);
                    console.log('Dose isosurfaces not available. Generate them in ShioRIS3 using "3D Isosurface" button.');
                }
            }

            // Show slice controls
            const sliceControls = document.getElementById('slice-controls');
            if (sliceControls) {
                sliceControls.style.display = 'block';
            }

            this.hideLoading();
            console.log('Patient loaded successfully');
            alert('DICOM volume loaded successfully!\n\nRotate: Mouse drag\nZoom: Mouse wheel\nPan: Right-click drag\nUse sliders to navigate slices');
        } catch (error) {
            console.error('Failed to load patient:', error);
            this.hideLoading();
            alert(`Failed to load patient data: ${error.message}\n\nMake sure you have loaded a DICOM volume in ShioRIS3.`);
        }
    }

    setupEventListeners() {
        // Load patient button
        const loadButton = document.getElementById('load-patient-btn');
        loadButton.addEventListener('click', () => {
            const select = document.getElementById('patient-select');
            const patientId = select.value;
            if (patientId) {
                this.loadPatient(patientId);
            } else {
                alert('Please select a patient first');
            }
        });

        // Display option checkboxes
        document.getElementById('show-ct').addEventListener('change', (e) => {
            this.volumeRenderer.setVolumeVisible(e.target.checked);
        });

        document.getElementById('show-structures').addEventListener('change', async (e) => {
            if (e.target.checked && this.currentPatientId) {
                try {
                    // Use simplify=3 for VR performance
                    const structureData = await this.apiClient.getStructures(this.currentPatientId, 3);
                    if (structureData.status === 'success') {
                        this.volumeRenderer.addStructures(structureData);
                        console.log(`RT Structures loaded: ${structureData.totalPoints} points (${structureData.simplification}x simplification)`);
                    }
                } catch (error) {
                    console.error('Failed to load structures:', error);
                    alert('Failed to load RT Structure data.\n\nMake sure you have loaded an RT Structure file in ShioRIS3.');
                    e.target.checked = false;
                    return;
                }
            }
            this.volumeRenderer.setStructuresVisible(e.target.checked);
        });

        document.getElementById('show-dose').addEventListener('change', async (e) => {
            if (e.target.checked && this.currentPatientId) {
                try {
                    const doseData = await this.apiClient.getDose(this.currentPatientId);
                    if (doseData.status === 'success' && doseData.isosurfaceCount > 0) {
                        this.volumeRenderer.addDose(doseData);
                        console.log(`Dose data loaded: ${doseData.isosurfaceCount} isosurfaces`);
                    } else {
                        console.log('No dose isosurfaces available');
                        alert('No dose isosurfaces available.\n\nPlease generate them in ShioRIS3:\n1. Load dose data\n2. Click "3D Isosurface" button');
                        e.target.checked = false;
                        return;
                    }
                } catch (error) {
                    console.error('Failed to load dose:', error);
                    alert('Failed to load dose data.\n\nMake sure you have:\n1. Loaded dose in ShioRIS3\n2. Generated isosurfaces using "3D Isosurface" button');
                    e.target.checked = false;
                    return;
                }
            }
            this.volumeRenderer.setDoseVisible(e.target.checked);
        });

        // Slice position sliders with debouncing to prevent stuttering
        const axialSlider = document.getElementById('axial-slider');
        const sagittalSlider = document.getElementById('sagittal-slider');
        const coronalSlider = document.getElementById('coronal-slider');

        // Axial slice with debounce
        axialSlider.addEventListener('input', (e) => {
            const position = parseInt(e.target.value) / 100;
            document.getElementById('axial-value').textContent = `${e.target.value}%`;

            // Update internal position immediately (for future use)
            this.volumeRenderer.axialPosition = position;

            // Debounce image loading
            clearTimeout(this.axialDebounceTimer);
            this.axialDebounceTimer = setTimeout(() => {
                this.volumeRenderer.setAxialPosition(position);
            }, this.sliceDebounceDelay);
        });

        // Sagittal slice with debounce
        sagittalSlider.addEventListener('input', (e) => {
            const position = parseInt(e.target.value) / 100;
            document.getElementById('sagittal-value').textContent = `${e.target.value}%`;

            // Update internal position immediately (for future use)
            this.volumeRenderer.sagittalPosition = position;

            // Debounce image loading
            clearTimeout(this.sagittalDebounceTimer);
            this.sagittalDebounceTimer = setTimeout(() => {
                this.volumeRenderer.setSagittalPosition(position);
            }, this.sliceDebounceDelay);
        });

        // Coronal slice with debounce
        coronalSlider.addEventListener('input', (e) => {
            const position = parseInt(e.target.value) / 100;
            document.getElementById('coronal-value').textContent = `${e.target.value}%`;

            // Update internal position immediately (for future use)
            this.volumeRenderer.coronalPosition = position;

            // Debounce image loading
            clearTimeout(this.coronalDebounceTimer);
            this.coronalDebounceTimer = setTimeout(() => {
                this.volumeRenderer.setCoronalPosition(position);
            }, this.sliceDebounceDelay);
        });

        console.log('Event listeners configured');
    }

    showLoading(message = 'Loading...') {
        const loadingElement = document.getElementById('loading');
        if (loadingElement) {
            const messageElement = loadingElement.querySelector('p');
            if (messageElement) {
                messageElement.textContent = message;
            }
            loadingElement.classList.remove('hidden');
        }
    }

    hideLoading() {
        const loadingElement = document.getElementById('loading');
        if (loadingElement) {
            loadingElement.classList.add('hidden');
        }
    }

    async loadBuildInfo() {
        if (!this.buildVersionElement || !this.buildStatusElement) {
            return;
        }

        const versionUrl = `version.json?ts=${Date.now()}`;

        try {
            const response = await fetch(versionUrl, { cache: 'no-store' });
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }

            const info = await response.json();
            this.buildInfo = info;
            window.__SHIORIS3_BUILD__ = info;
            console.log('Loaded build info:', info);
            if (this.diagnosticsRoot) {
                this.diagnosticsRoot.dataset.state = 'ok';
            }
            this.updateBuildInfoUI(info, false);
        } catch (error) {
            console.warn('Failed to load version.json', error);
            const fallbackInfo = {
                version: 'unknown',
                builtAt: document.lastModified || new Date().toISOString(),
                notes: 'version.json not reachable'
            };
            if (this.diagnosticsRoot) {
                this.diagnosticsRoot.dataset.state = 'error';
            }
            this.updateBuildInfoUI(fallbackInfo, true);
        }
    }

    updateBuildInfoUI(buildInfo, isFallback = false) {
        if (!this.buildVersionElement || !this.buildStatusElement) {
            return;
        }

        const label = buildInfo?.version || 'unknown';
        this.buildVersionElement.textContent = `Webビルド: ${label}`;

        const messageParts = [];
        if (buildInfo?.builtAt) {
            messageParts.push(`更新: ${buildInfo.builtAt}`);
        }
        if (buildInfo?.commit) {
            messageParts.push(`commit: ${buildInfo.commit}`);
        }
        if (buildInfo?.notes) {
            messageParts.push(buildInfo.notes);
        }
        if (isFallback) {
            messageParts.push('※ version.jsonの読み込みに失敗しました');
        }

        if (!window.isSecureContext) {
            messageParts.push('HTTPSでアクセスしているか確認してください');
        }

        const previousVersion = window.localStorage ? window.localStorage.getItem(this.localStorageVersionKey) : null;
        if (previousVersion && previousVersion !== label) {
            messageParts.push(`前回: ${previousVersion}`);
        }
        if (window.localStorage) {
            window.localStorage.setItem(this.localStorageVersionKey, label);
        }

        this.buildStatusElement.textContent = messageParts.join(' / ') || '---';

        if (this.cacheBypassNoteElement) {
            this.cacheBypassNoteElement.innerHTML = 'Vision Proで表示されているバージョン番号が上記と異なる場合、下のボタンでキャッシュを削除して再読み込みしてください。<strong>HTTPS未設定のままではVision Proで没入表示できません。</strong>';
        }
    }

    setupCacheBypassButton() {
        if (!this.cacheBypassButton) {
            return;
        }

        this.cacheBypassButton.disabled = false;
        this.cacheBypassButton.addEventListener('click', () => this.forceCacheBypassReload());
    }

    async forceCacheBypassReload() {
        if (!this.cacheBypassButton) {
            return;
        }

        this.cacheBypassButton.disabled = true;
        const originalLabel = this.cacheBypassButton.textContent;
        this.cacheBypassButton.textContent = 'キャッシュ削除中...';

        try {
            if ('caches' in window && typeof window.caches.keys === 'function') {
                const cacheKeys = await window.caches.keys();
                await Promise.all(cacheKeys.map((key) => window.caches.delete(key)));
            }
            if (window.localStorage) {
                window.localStorage.removeItem(this.localStorageVersionKey);
            }
            if (window.sessionStorage) {
                window.sessionStorage.clear();
            }
            if (this.buildStatusElement) {
                this.buildStatusElement.textContent = 'ブラウザキャッシュを削除しました。最新データで再接続します。';
            }
        } catch (error) {
            console.warn('Failed to clear caches', error);
            if (this.buildStatusElement) {
                this.buildStatusElement.textContent = `キャッシュ削除に失敗しました: ${error.message}`;
            }
        } finally {
            const reloadUrl = `${window.location.origin}${window.location.pathname}?cacheBust=${Date.now()}`;
            setTimeout(() => {
                window.location.replace(reloadUrl);
            }, 500);
            this.cacheBypassButton.textContent = originalLabel;
        }
    }
}

// Initialize the application when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new ShioRIS3Viewer();
    });
} else {
    new ShioRIS3Viewer();
}
