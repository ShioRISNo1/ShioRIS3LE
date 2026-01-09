/**
 * Volume Renderer for CT/MRI data visualization
 * Uses Three.js for 3D rendering
 * VERSION: v11-2025-11-15 - Match OpenGL: transparent+depthWrite+alphaTest=0.01
 */
console.log('📦 LOADING volume_renderer.js v11 - OpenGL-compatible: transparent=true, depthWrite=true, alphaTest=0.01');

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class VolumeRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.volumeMesh = null;
        this.structureMeshes = [];
        this.doseMeshes = [];

        // Orthogonal slices
        this.axialSlice = null;
        this.sagittalSlice = null;
        this.coronalSlice = null;

        // Slice positions (0.0 to 1.0)
        this.axialPosition = 0.5;
        this.sagittalPosition = 0.5;
        this.coronalPosition = 0.5;

        // Volume metadata
        this.volumeData = null;
        this.apiClient = null;
        this.patientId = null;

        // Window/Level settings (matching MainWindow defaults)
        this.window = 256.0;
        this.level = 128.0;

        this.init();
    }

    init() {
        // Create scene
        this.scene = new THREE.Scene();
        // Default dark background for normal browser use
        // Will be set to null (transparent) when entering VR/AR mode
        this.defaultBackgroundColor = 0x1a1a2e;
        this.scene.background = new THREE.Color(this.defaultBackgroundColor);

        // Group containing all CT/Structure/Dose meshes so we can reposition
        // them in XR mode without affecting lights or the camera
        this.sceneContent = new THREE.Group();
        this.sceneContent.name = 'VolumeContentGroup';
        this.scene.add(this.sceneContent);

        // Store default transform for resetting after XR sessions
        this.defaultContentPosition = new THREE.Vector3(0, 0, 0);
        this.defaultContentScale = new THREE.Vector3(1, 1, 1);
        this.sceneContent.position.copy(this.defaultContentPosition);
        this.sceneContent.scale.copy(this.defaultContentScale);

        // Create camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            this.canvas.clientWidth / this.canvas.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 0, 2);
        this.camera.lookAt(0, 0, 0);  // Look at origin

        // Create renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: true,
            logarithmicDepthBuffer: true  // Better depth precision for overlapping slices
        });
        this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.sortObjects = true;  // Enable automatic transparency sorting
        this.renderer.setClearColor(0x000000, 0);  // Ensure transparent clear
        this.renderer.domElement.style.backgroundColor = 'transparent';
        this.canvas.style.backgroundColor = 'transparent';
        this.renderer.xr.enabled = true;
        // Reference space is set by VRController based on session mode (AR='local', VR='local-floor')

        // XR placement presets (values are tuned for Vision Pro AR view)
        // Y=0.5 (50cm) places content lower for more stable viewing
        // Lower position reduces apparent shake in upper parts due to rotation center distance
        this.xrARPosition = new THREE.Vector3(0, 0.5, -1.8);
        this.xrVRPosition = new THREE.Vector3(0, 0.5, -1.8);
        this.xrARScale = 1.0;  // Reduced from 1.2 to minimize apparent shake
        this.xrVRScale = 1.0;

        // Create controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.target.set(0, 0, 0);  // Rotate around origin (volume center)
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.minDistance = 0.5;
        this.controls.maxDistance = 10;

        // Add lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        this.scene.add(directionalLight);

        // Add axes helper (size 0.5 for normalized coordinates)
        this.axesHelper = new THREE.AxesHelper(0.5);
        this.sceneContent.add(this.axesHelper);

        // Add grid helper (size 1.0 for normalized coordinates -0.5 to +0.5)
        this.gridHelper = new THREE.GridHelper(1.0, 10, 0x667eea, 0x444466);
        this.sceneContent.add(this.gridHelper);

        console.log('🔧 VolumeRenderer: Grid size 1.0, Axes 0.5, Camera at z=2 [2025-11-15 UPDATE]');

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());

        // Start animation loop
        this.animate();

        console.log('VolumeRenderer initialized');
    }

    /**
     * Create a simple box mesh representing CT volume
     * This is a placeholder - will be replaced with actual volume rendering
     */
    createDemoVolume(width = 100, height = 100, depth = 100) {
        // Remove existing volume
        if (this.volumeMesh) {
            this.sceneContent.remove(this.volumeMesh);
        }

        // Create a semi-transparent box to represent the volume
        const geometry = new THREE.BoxGeometry(width, height, depth);
        const material = new THREE.MeshPhongMaterial({
            color: 0x4488ff,
            transparent: true,
            opacity: 0.3,
            wireframe: false
        });

        this.volumeMesh = new THREE.Mesh(geometry, material);
        this.sceneContent.add(this.volumeMesh);

        // Add wireframe
        const wireframeGeometry = new THREE.EdgesGeometry(geometry);
        const wireframeMaterial = new THREE.LineBasicMaterial({ color: 0x88ccff });
        const wireframe = new THREE.LineSegments(wireframeGeometry, wireframeMaterial);
        this.volumeMesh.add(wireframe);

        console.log('Demo volume created:', width, height, depth);
    }

    /**
     * Synchronize Window/Level settings from MainWindow via API
     */
    async syncWindowLevelFromMainWindow() {
        if (!this.apiClient) return;

        try {
            const response = await fetch('/api/window-level');
            if (!response.ok) {
                console.warn('Failed to fetch window/level from API');
                return;
            }

            const data = await response.json();
            if (data.status === 'success') {
                const newWindow = data.window;
                const newLevel = data.level;

                // Only update if values changed
                if (newWindow !== this.window || newLevel !== this.level) {
                    console.log(`Syncing Window/Level from MainWindow: w=${newWindow}, l=${newLevel}`);
                    await this.setWindowLevel(newWindow, newLevel);
                }
            }
        } catch (error) {
            console.error('Error syncing window/level:', error);
        }
    }

    /**
     * Create volume from actual DICOM data with 3 orthogonal slices
     * @param {Object} volumeData - Volume metadata and voxel data
     * @param {Object} apiClient - API client for loading slices
     * @param {string} patientId - Patient ID
     */
    async createVolumeFromData(volumeData, apiClient, patientId) {
        console.log('Creating orthogonal slices from data:', volumeData);

        if (!volumeData.volume) {
            console.error('No volume metadata - cannot create slices');
            return;
        }

        // Store for later use
        this.volumeData = volumeData;
        this.apiClient = apiClient;
        this.patientId = patientId;

        const { width, height, depth, spacingX, spacingY, spacingZ } = volumeData.volume;
        console.log(`Volume dimensions: ${width}x${height}x${depth}, spacing: ${spacingX}x${spacingY}x${spacingZ}`);

        // Remove existing slices
        if (this.axialSlice) this.sceneContent.remove(this.axialSlice);
        if (this.sagittalSlice) this.sceneContent.remove(this.sagittalSlice);
        if (this.coronalSlice) this.sceneContent.remove(this.coronalSlice);

        // Calculate physical dimensions in mm (matching OpenGL implementation)
        this.px_mm = width * spacingX;
        this.py_mm = height * spacingY;
        this.pz_mm = depth * spacingZ;

        // Normalize by largest dimension (matching OpenGL implementation)
        const maxDim = Math.max(this.px_mm, this.py_mm, this.pz_mm);
        this.sx = this.px_mm / maxDim;
        this.sy = this.py_mm / maxDim;
        this.sz = this.pz_mm / maxDim;

        console.log(`Physical dimensions (mm): X=${this.px_mm}, Y=${this.py_mm}, Z=${this.pz_mm}`);
        console.log(`Normalized scale: sx=${this.sx}, sy=${this.sy}, sz=${this.sz}`);

        // Sync Window/Level from MainWindow AFTER physical dimensions are calculated
        await this.syncWindowLevelFromMainWindow();

        // Create the three orthogonal slices at initial positions
        console.log('Creating axial slice...');
        await this.updateAxialSlice();
        console.log('Creating sagittal slice...');
        await this.updateSagittalSlice();
        console.log('Creating coronal slice...');
        await this.updateCoronalSlice();

        console.log('Orthogonal slices created');
    }

    /**
     * Convert mm coordinates to Three.js GL coordinates
     * Matches OpenGL implementation: x = (-x_mm / px_mm) * sx, etc.
     */
    mmToGL(x_mm, y_mm, z_mm) {
        return new THREE.Vector3(
            (-x_mm / this.px_mm) * this.sx,
            (y_mm / this.py_mm) * this.sy,
            (z_mm / this.pz_mm) * this.sz
        );
    }

    /**
     * Update axial slice at current position
     */
    async updateAxialSlice() {
        if (!this.volumeData) return;

        const { depth } = this.volumeData.volume;
        const sliceIndex = Math.floor(this.axialPosition * (depth - 1));
        // Calculate mm position: (index/(depth-1) - 0.5) * pz_mm
        const axZmm = (depth > 1) ? ((sliceIndex / (depth - 1)) - 0.5) * this.pz_mm : 0.0;

        try {
            console.log(`Updating axial slice: position=${this.axialPosition.toFixed(3)}, index=${sliceIndex}/${depth - 1}, axZmm=${axZmm.toFixed(2)}`);

            const sliceURL = await this.apiClient.getSliceImageURL(this.patientId, 'axial', sliceIndex, this.window, this.level);
            const texture = await this.loadTextureFromURL(sliceURL);

            // No flip needed - C++ backend handles Axial flipping
            texture.flipY = false;
            texture.flipX = false;
            texture.needsUpdate = true;
            console.log('🔧 Axial texture: no flip (C++ backend handles flipping)');

            // Remove old slice
            if (this.axialSlice) {
                this.sceneContent.remove(this.axialSlice);
            }

            // Create axial slice matching OpenGL implementation
            // OpenGL vertices (with mirrored texture):
            // v0 = mmToGL(+px_mm/2, +py_mm/2, axZmm), tex(0,0)
            // v1 = mmToGL(-px_mm/2, +py_mm/2, axZmm), tex(1,0)
            // v2 = mmToGL(-px_mm/2, -py_mm/2, axZmm), tex(1,1)
            // v3 = mmToGL(+px_mm/2, -py_mm/2, axZmm), tex(0,1)

            const v0 = this.mmToGL(this.px_mm / 2, this.py_mm / 2, axZmm);
            const v1 = this.mmToGL(-this.px_mm / 2, this.py_mm / 2, axZmm);
            const v2 = this.mmToGL(-this.px_mm / 2, -this.py_mm / 2, axZmm);
            const v3 = this.mmToGL(this.px_mm / 2, -this.py_mm / 2, axZmm);

            const geometry = new THREE.BufferGeometry();
            const vertices = new Float32Array([
                v0.x, v0.y, v0.z,
                v1.x, v1.y, v1.z,
                v2.x, v2.y, v2.z,
                v0.x, v0.y, v0.z,
                v2.x, v2.y, v2.z,
                v3.x, v3.y, v3.z
            ]);
            // UV coordinates - standard mapping (C++ backend handles flipping)
            const uvs = new Float32Array([
                0, 0,
                1, 0,
                1, 1,
                0, 0,
                1, 1,
                0, 1
            ]);
            console.log('🔧 Axial UV: standard [(0,0), (1,0), (1,1), (0,1)] (C++ backend handles flipping)');
            geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));

            const material = new THREE.MeshBasicMaterial({
                map: texture,
                transparent: true,   // Enable alpha blending (like GL_BLEND)
                opacity: 1.0,
                alphaTest: 0.0,      // Match Desktop OpenGL: glAlphaFunc(GL_GREATER, 0.0f)
                side: THREE.DoubleSide,
                depthWrite: false,   // Don't write transparent pixels to depth buffer
                depthTest: true,
                blending: THREE.NormalBlending  // Standard alpha blending
            });

            this.axialSlice = new THREE.Mesh(geometry, material);
            this.axialSlice.renderOrder = 1;  // Axial renders second
            this.sceneContent.add(this.axialSlice);

            console.log(`Axial slice updated successfully at axZmm=${axZmm.toFixed(2)}, renderOrder=1 (alphaTest=0.0, depthWrite=false)`);
        } catch (error) {
            console.error('Failed to update axial slice:', error);
        }
    }

    /**
     * Update sagittal slice at current position
     */
    async updateSagittalSlice() {
        if (!this.volumeData) return;

        const { width } = this.volumeData.volume;
        const sliceIndex = Math.floor(this.sagittalPosition * (width - 1));
        // Calculate mm position: (index/(width-1) - 0.5) * px_mm
        const sagXmm = (width > 1) ? ((sliceIndex / (width - 1)) - 0.5) * this.px_mm : 0.0;

        try {
            console.log(`Updating sagittal slice: position=${this.sagittalPosition.toFixed(3)}, index=${sliceIndex}/${width - 1}, sagXmm=${sagXmm.toFixed(2)}`);

            const sliceURL = await this.apiClient.getSliceImageURL(this.patientId, 'sagittal', sliceIndex, this.window, this.level);
            const texture = await this.loadTextureFromURL(sliceURL);

            // Remove old slice
            if (this.sagittalSlice) {
                this.sceneContent.remove(this.sagittalSlice);
            }

            // Create sagittal slice matching OpenGL implementation
            // OpenGL vertices (no texture mirroring):
            // v0 = mmToGL(sagXmm, +py_mm/2, +pz_mm/2), tex(0,0)
            // v1 = mmToGL(sagXmm, -py_mm/2, +pz_mm/2), tex(1,0)
            // v2 = mmToGL(sagXmm, -py_mm/2, -pz_mm/2), tex(1,1)
            // v3 = mmToGL(sagXmm, +py_mm/2, -pz_mm/2), tex(0,1)

            const v0 = this.mmToGL(sagXmm, this.py_mm / 2, this.pz_mm / 2);
            const v1 = this.mmToGL(sagXmm, -this.py_mm / 2, this.pz_mm / 2);
            const v2 = this.mmToGL(sagXmm, -this.py_mm / 2, -this.pz_mm / 2);
            const v3 = this.mmToGL(sagXmm, this.py_mm / 2, -this.pz_mm / 2);

            const geometry = new THREE.BufferGeometry();
            const vertices = new Float32Array([
                v0.x, v0.y, v0.z,
                v1.x, v1.y, v1.z,
                v2.x, v2.y, v2.z,
                v0.x, v0.y, v0.z,
                v2.x, v2.y, v2.z,
                v3.x, v3.y, v3.z
            ]);
            const uvs = new Float32Array([
                0, 0,
                1, 0,
                1, 1,
                0, 0,
                1, 1,
                0, 1
            ]);
            geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));

            const material = new THREE.MeshBasicMaterial({
                map: texture,
                transparent: true,   // Enable alpha blending (like GL_BLEND)
                opacity: 1.0,
                alphaTest: 0.0,      // Match Desktop OpenGL: glAlphaFunc(GL_GREATER, 0.0f)
                side: THREE.DoubleSide,
                depthWrite: false,   // Don't write transparent pixels to depth buffer
                depthTest: true,
                blending: THREE.NormalBlending  // Standard alpha blending
            });

            this.sagittalSlice = new THREE.Mesh(geometry, material);
            this.sagittalSlice.renderOrder = 2;  // Sagittal renders third
            this.sceneContent.add(this.sagittalSlice);

            console.log(`Sagittal slice updated successfully at sagXmm=${sagXmm.toFixed(2)}, renderOrder=2 (alphaTest=0.0, depthWrite=false)`);
        } catch (error) {
            console.error('Failed to update sagittal slice:', error);
        }
    }

    /**
     * Update coronal slice at current position
     */
    async updateCoronalSlice() {
        if (!this.volumeData) return;

        const { height } = this.volumeData.volume;
        const sliceIndex = Math.floor(this.coronalPosition * (height - 1));
        // Calculate mm position: ((height-1-index)/(height-1) - 0.5) * py_mm
        // Note: index is inverted in OpenGL implementation
        const corYmm = (height > 1) ? (((height - 1 - sliceIndex) / (height - 1)) - 0.5) * this.py_mm : 0.0;

        try {
            console.log(`Updating coronal slice: position=${this.coronalPosition.toFixed(3)}, index=${sliceIndex}/${height - 1}, corYmm=${corYmm.toFixed(2)}`);

            const sliceURL = await this.apiClient.getSliceImageURL(this.patientId, 'coronal', sliceIndex, this.window, this.level);
            const texture = await this.loadTextureFromURL(sliceURL);

            // Remove old slice
            if (this.coronalSlice) {
                this.sceneContent.remove(this.coronalSlice);
            }

            // Create coronal slice matching OpenGL implementation
            // OpenGL vertices (with vertically mirrored texture):
            // v0 = mmToGL(-px_mm/2, corYmm, -pz_mm/2), tex(0,0)
            // v1 = mmToGL(+px_mm/2, corYmm, -pz_mm/2), tex(1,0)
            // v2 = mmToGL(+px_mm/2, corYmm, +pz_mm/2), tex(1,1)
            // v3 = mmToGL(-px_mm/2, corYmm, +pz_mm/2), tex(0,1)

            const v0 = this.mmToGL(-this.px_mm / 2, corYmm, -this.pz_mm / 2);
            const v1 = this.mmToGL(this.px_mm / 2, corYmm, -this.pz_mm / 2);
            const v2 = this.mmToGL(this.px_mm / 2, corYmm, this.pz_mm / 2);
            const v3 = this.mmToGL(-this.px_mm / 2, corYmm, this.pz_mm / 2);

            const geometry = new THREE.BufferGeometry();
            const vertices = new Float32Array([
                v0.x, v0.y, v0.z,
                v1.x, v1.y, v1.z,
                v2.x, v2.y, v2.z,
                v0.x, v0.y, v0.z,
                v2.x, v2.y, v2.z,
                v3.x, v3.y, v3.z
            ]);
            const uvs = new Float32Array([
                0, 0,
                1, 0,
                1, 1,
                0, 0,
                1, 1,
                0, 1
            ]);
            geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));

            const material = new THREE.MeshBasicMaterial({
                map: texture,
                transparent: true,   // Enable alpha blending (like GL_BLEND)
                opacity: 1.0,
                alphaTest: 0.0,      // Match Desktop OpenGL: glAlphaFunc(GL_GREATER, 0.0f)
                side: THREE.DoubleSide,
                depthWrite: false,   // Don't write transparent pixels to depth buffer
                depthTest: true,
                blending: THREE.NormalBlending  // Standard alpha blending
            });

            this.coronalSlice = new THREE.Mesh(geometry, material);
            this.coronalSlice.renderOrder = 3;  // Coronal renders last
            this.sceneContent.add(this.coronalSlice);

            console.log(`Coronal slice updated successfully at corYmm=${corYmm.toFixed(2)}, renderOrder=3 (alphaTest=0.0, depthWrite=false)`);
        } catch (error) {
            console.error('Failed to update coronal slice:', error);
        }
    }

    /**
     * Set axial slice position (0.0 to 1.0)
     */
    async setAxialPosition(position) {
        this.axialPosition = Math.max(0, Math.min(1, position));
        await this.updateAxialSlice();
    }

    /**
     * Set sagittal slice position (0.0 to 1.0)
     */
    async setSagittalPosition(position) {
        this.sagittalPosition = Math.max(0, Math.min(1, position));
        await this.updateSagittalSlice();
    }

    /**
     * Set coronal slice position (0.0 to 1.0)
     */
    async setCoronalPosition(position) {
        this.coronalPosition = Math.max(0, Math.min(1, position));
        await this.updateCoronalSlice();
    }

    /**
     * Set window/level and refresh all slices
     * Called from MainWindow when user adjusts Window/Level
     */
    async setWindowLevel(window, level) {
        if (window === undefined || window === null) window = 256.0;
        if (level === undefined || level === null) level = 128.0;

        this.window = window;
        this.level = level;

        console.log(`VolumeRenderer: Window/Level updated to w=${window}, l=${level}`);

        // Refresh all slices with new window/level
        if (this.volumeData) {
            await Promise.all([
                this.updateAxialSlice(),
                this.updateSagittalSlice(),
                this.updateCoronalSlice()
            ]);
            console.log('All slices refreshed with new Window/Level');
        }
    }

    /**
     * Load a texture from a URL (promise-based)
     */
    loadTextureFromURL(url) {
        return new Promise((resolve, reject) => {
            const loader = new THREE.TextureLoader();
            loader.load(
                url,
                (texture) => {
                    texture.minFilter = THREE.LinearFilter;
                    texture.magFilter = THREE.LinearFilter;
                    texture.format = THREE.RGBAFormat;  // Explicitly specify RGBA format
                    texture.colorSpace = THREE.SRGBColorSpace;  // Proper color space for PNG
                    texture.needsUpdate = true;
                    resolve(texture);
                },
                undefined,
                (error) => {
                    reject(error);
                }
            );
        });
    }

    /**
     * Add structure contours to the scene
     */
    addStructures(structureData) {
        // Clear existing structures
        this.structureMeshes.forEach(mesh => this.sceneContent.remove(mesh));
        this.structureMeshes = [];

        if (!structureData || !structureData.structures) {
            console.log('No structure data available');
            return;
        }

        console.log('Adding structures:', structureData);

        // Create a group for all structures
        const structureGroup = new THREE.Group();

        console.log(`Volume data available: ${this.volumeData !== null}`);

        if (!this.volumeData) {
            console.error('Cannot add structures without volume data');
            return;
        }

        // Process each structure (ROI)
        for (const structure of structureData.structures) {
            if (!structure.contours) continue;

            console.log(`Processing structure: ${structure.name}, contours: ${structure.contours.length}`);

            // Process each contour
            for (const contour of structure.contours) {
                if (!contour.points || contour.points.length < 2) continue;

                // Create line geometry from points
                // Structure coordinates are in DICOM patient mm coordinates
                // Use mmToGL to convert, matching OpenGL implementation
                const points = [];
                for (const point of contour.points) {
                    const glPos = this.mmToGL(point.x, point.y, point.z);
                    points.push(glPos);
                }

                // Close the contour if it's not already closed
                if (points.length > 0) {
                    const firstPoint = points[0];
                    const lastPoint = points[points.length - 1];
                    if (!firstPoint.equals(lastPoint)) {
                        points.push(firstPoint.clone());
                    }
                }

                // Create line geometry
                const geometry = new THREE.BufferGeometry().setFromPoints(points);

                // Create line material with the contour's color
                const color = new THREE.Color(
                    contour.color.r / 255,
                    contour.color.g / 255,
                    contour.color.b / 255
                );

                const material = new THREE.LineBasicMaterial({
                    color: color,
                    linewidth: 2,
                    opacity: contour.color.a / 255,
                    transparent: true
                });

                // Create line
                const line = new THREE.Line(geometry, material);
                structureGroup.add(line);
            }
        }

        // Add structure group to scene
        this.sceneContent.add(structureGroup);
        this.structureMeshes.push(structureGroup);

        console.log(`Added ${structureData.totalContours} contours from ${structureData.structureCount} structures`);
    }

    /**
     * Add dose distribution to the scene
     */
    addDose(doseData) {
        // Clear existing dose meshes
        this.doseMeshes.forEach(mesh => this.sceneContent.remove(mesh));
        this.doseMeshes = [];

        if (!doseData || !doseData.isosurfaces || doseData.isosurfaces.length === 0) {
            console.log('No dose isosurfaces to display');
            return;
        }

        console.log(`Adding ${doseData.isosurfaceCount} dose isosurfaces with ${doseData.totalTriangles} total triangles`);

        if (!this.volumeData) {
            console.error('Cannot add dose isosurfaces without volume data');
            return;
        }

        // Process each isosurface
        for (const isosurface of doseData.isosurfaces) {
            if (!isosurface.vertices || !isosurface.normals || isosurface.triangleCount === 0) continue;

            console.log(`Processing isosurface with ${isosurface.triangleCount} triangles, opacity: ${isosurface.opacity}`);

            // Create geometry from flat arrays
            const vertices = [];
            const normals = [];

            // Process vertices (flat array: [x0,y0,z0, x1,y1,z1, ...])
            for (let i = 0; i < isosurface.vertices.length; i += 3) {
                const x_mm = isosurface.vertices[i];
                const y_mm = isosurface.vertices[i + 1];
                const z_mm = isosurface.vertices[i + 2];

                // Convert from patient mm coordinates to GL coordinates
                const glPos = this.mmToGL(x_mm, y_mm, z_mm);
                vertices.push(glPos.x, glPos.y, glPos.z);
            }

            // Process normals (flat array: [nx0,ny0,nz0, nx1,ny1,nz1, ...])
            // Each triangle has 1 normal, replicate it for all 3 vertices
            for (let i = 0; i < isosurface.normals.length; i += 3) {
                const nx = isosurface.normals[i];
                const ny = isosurface.normals[i + 1];
                const nz = isosurface.normals[i + 2];

                // Add same normal for all 3 vertices of triangle
                for (let j = 0; j < 3; j++) {
                    normals.push(nx, ny, nz);
                }
            }

            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices), 3));
            geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(normals), 3));

            // Create material with color and opacity
            const color = new THREE.Color(
                isosurface.color.r / 255,
                isosurface.color.g / 255,
                isosurface.color.b / 255
            );

            const material = new THREE.MeshPhongMaterial({
                color: color,
                opacity: isosurface.opacity,
                transparent: true,
                side: THREE.DoubleSide,
                depthWrite: false,  // Don't write to depth buffer for proper transparency
                depthTest: true,
                blending: THREE.NormalBlending
            });

            // Create mesh
            const mesh = new THREE.Mesh(geometry, material);
            mesh.renderOrder = 0;  // Render dose before slices
            this.sceneContent.add(mesh);
            this.doseMeshes.push(mesh);

            console.log(`Added dose isosurface mesh with ${isosurface.triangleCount} triangles`);
        }

        console.log(`Total dose meshes added: ${this.doseMeshes.length}`);
    }

    /**
     * Set visibility of volume
     */
    setVolumeVisible(visible) {
        if (this.volumeMesh) {
            this.volumeMesh.visible = visible;
        }
    }

    /**
     * Set visibility of structures
     */
    setStructuresVisible(visible) {
        this.structureMeshes.forEach(mesh => {
            mesh.visible = visible;
        });
    }

    /**
     * Set visibility of dose
     */
    setDoseVisible(visible) {
        this.doseMeshes.forEach(mesh => {
            mesh.visible = visible;
        });
    }

    /**
     * Animation loop
     */
    animate() {
        // Use XR-compatible animation loop
        this.renderer.setAnimationLoop(() => {
            this.controls.update();
            this.renderer.render(this.scene, this.camera);
        });
    }

    /**
     * Handle window resize
     */
    onWindowResize() {
        const width = this.canvas.clientWidth;
        const height = this.canvas.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);
    }

    /**
     * Get the WebXR-enabled renderer
     */
    getRenderer() {
        return this.renderer;
    }

    /**
     * Get the scene
     */
    getScene() {
        return this.scene;
    }

    /**
     * Configure placement/background when entering XR (Vision Pro) sessions
     */
    enterXRMode(sessionMode = 'immersive-ar') {
        const isAR = sessionMode === 'immersive-ar';
        const targetPosition = isAR ? this.xrARPosition : this.xrVRPosition;
        const targetScale = isAR ? this.xrARScale : this.xrVRScale;

        this.sceneContent.position.copy(targetPosition);
        this.sceneContent.scale.set(targetScale, targetScale, targetScale);

        // Set background based on session mode
        if (isAR) {
            // AR mode: transparent background for passthrough (environmentBlendMode='alpha-blend')
            this.scene.background = null;
            this.renderer.setClearColor(0x000000, 0);  // alpha=0 (fully transparent)
            this.renderer.domElement.style.backgroundColor = 'transparent';
        } else {
            // VR mode: opaque background (environmentBlendMode='opaque', no passthrough)
            this.scene.background = new THREE.Color(0x1a1a2e);  // Dark blue background
            this.renderer.setClearColor(0x1a1a2e, 1);  // alpha=1 (fully opaque)
            this.renderer.domElement.style.backgroundColor = '#1a1a2e';
        }

        // Hide grid and axes helpers in XR mode (they block passthrough)
        if (this.gridHelper) this.gridHelper.visible = false;
        if (this.axesHelper) this.axesHelper.visible = false;

        // Disable OrbitControls in XR mode (they interfere with head tracking)
        if (this.controls) this.controls.enabled = false;

        console.log(`VolumeRenderer: enterXRMode -> mode=${sessionMode}, position=${targetPosition.toArray()}, scale=${targetScale}, background=${isAR ? 'transparent' : 'opaque'}`);
    }

    /**
     * Restore placement/background after leaving XR
     */
    exitXRMode() {
        this.sceneContent.position.copy(this.defaultContentPosition);
        this.sceneContent.scale.copy(this.defaultContentScale);
        this.scene.background = new THREE.Color(this.defaultBackgroundColor);

        // Restore grid and axes helpers visibility
        if (this.gridHelper) this.gridHelper.visible = true;
        if (this.axesHelper) this.axesHelper.visible = true;

        // Re-enable OrbitControls for desktop mode
        if (this.controls) this.controls.enabled = true;

        console.log('VolumeRenderer: exitXRMode -> restored desktop placement');
    }

    /**
     * Get the camera
     */
    getCamera() {
        return this.camera;
    }

    /**
     * Clean up resources
     */
    dispose() {
        if (this.controls) {
            this.controls.dispose();
        }
        if (this.renderer) {
            this.renderer.dispose();
        }
    }
}
