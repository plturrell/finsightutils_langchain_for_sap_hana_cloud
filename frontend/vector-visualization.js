// SAP HANA Cloud LangChain Integration - Vector Visualization Component
// This script provides an interactive 3D visualization of vector embeddings

class VectorVisualization {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container element with ID '${containerId}' not found`);
            return;
        }
        
        this.options = Object.assign({
            height: '70vh',
            colorScheme: 'default',
            highContrast: false,
            showLabels: true,
            colorblindMode: false,
            pointSize: 5,
            queryPointSize: 8,
            maxPoints: 100,
            backgroundColor: '#f8f9fa',
            darkModeBackgroundColor: '#2a2a2a'
        }, options);
        
        // State
        this.points = [];
        this.queryPoint = null;
        this.selectedPointIndex = null;
        this.metadata = [];
        this.contents = [];
        this.similarities = [];
        
        // Color schemes
        this.colorSchemes = {
            default: {
                query: '#ff3e00',
                points: '#0070f3',
                selected: '#00c853',
                gradient: ['#0070f3', '#00c853', '#ffc107', '#ff3e00']
            },
            colorblind: {
                query: '#E69F00',
                points: '#56B4E9',
                selected: '#009E73',
                gradient: ['#56B4E9', '#009E73', '#F0E442', '#E69F00']
            },
            highContrast: {
                query: '#FFFFFF',
                points: '#0000FF',
                selected: '#00FF00',
                gradient: ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']
            }
        };
        
        this.initVisualization();
    }
    
    // Initialize the visualization
    initVisualization() {
        // Create container structure
        this.container.innerHTML = `
            <div class="vector-vis-container" style="position: relative; height: ${this.options.height}; width: 100%;">
                <div class="vector-vis-canvas" style="width: 100%; height: 100%; border-radius: 8px; overflow: hidden;"></div>
                <div class="vector-vis-controls" style="position: absolute; top: 10px; right: 10px; z-index: 100;">
                    <div class="btn-group-vertical">
                        <button class="btn btn-sm btn-light vector-vis-reset" title="Reset view">
                            <span class="visually-hidden">Reset view</span>
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                                <path d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2v1z"/>
                                <path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466z"/>
                            </svg>
                        </button>
                        <button class="btn btn-sm btn-light vector-vis-labels" title="Toggle labels">
                            <span class="visually-hidden">Toggle labels</span>
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                                <path d="M2 10h3a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1zm9-9h3a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-3a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1zm0 9h3a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-3a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1zM2 1h3a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1z"/>
                            </svg>
                        </button>
                    </div>
                </div>
                <div class="vector-vis-info" style="position: absolute; bottom: 0; left: 0; right: 0; background: rgba(255,255,255,0.8); padding: 10px; display: none; max-height: 30%; overflow-y: auto; border-top: 1px solid #ddd;">
                    <h6 class="vector-vis-info-title"></h6>
                    <p class="vector-vis-info-content"></p>
                    <div class="vector-vis-info-similarity"></div>
                </div>
            </div>
            <div class="vector-vis-legend mt-2 d-flex justify-content-center">
                <div class="d-flex align-items-center me-3">
                    <span class="vector-vis-legend-query me-1" style="display: inline-block; width: 12px; height: 12px; border-radius: 50%;"></span>
                    <small>Query</small>
                </div>
                <div class="d-flex align-items-center me-3">
                    <span class="vector-vis-legend-point me-1" style="display: inline-block; width: 12px; height: 12px; border-radius: 50%;"></span>
                    <small>Results</small>
                </div>
                <div class="d-flex align-items-center">
                    <span class="vector-vis-legend-selected me-1" style="display: inline-block; width: 12px; height: 12px; border-radius: 50%;"></span>
                    <small>Selected</small>
                </div>
            </div>
        `;
        
        // Get references to DOM elements
        this.canvasContainer = this.container.querySelector('.vector-vis-canvas');
        this.infoPanel = this.container.querySelector('.vector-vis-info');
        this.infoTitle = this.container.querySelector('.vector-vis-info-title');
        this.infoContent = this.container.querySelector('.vector-vis-info-content');
        this.infoSimilarity = this.container.querySelector('.vector-vis-info-similarity');
        this.resetButton = this.container.querySelector('.vector-vis-reset');
        this.labelsButton = this.container.querySelector('.vector-vis-labels');
        
        // Set legend colors
        this.updateColorScheme();
        
        // Initialize events
        this.resetButton.addEventListener('click', () => this.resetView());
        this.labelsButton.addEventListener('click', () => this.toggleLabels());
        
        // Create empty visualization
        this.createEmptyScene();
        
        // Setup for accessibility
        this.setupAccessibility();
    }
    
    // Create empty 3D scene
    createEmptyScene() {
        // If Three.js is not available, create a placeholder
        if (typeof THREE === 'undefined') {
            this.canvasContainer.innerHTML = `
                <div class="vector-vis-placeholder d-flex flex-column justify-content-center align-items-center h-100">
                    <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" fill="currentColor" class="bi bi-box mb-3" viewBox="0 0 16 16">
                        <path d="M8.186 1.113a.5.5 0 0 0-.372 0L1.846 3.5 8 5.961 14.154 3.5 8.186 1.113zM15 4.239l-6.5 2.6v7.922l6.5-2.6V4.24zM7.5 14.762V6.838L1 4.239v7.923l6.5 2.6zM7.443.184a1.5 1.5 0 0 1 1.114 0l7.129 2.852A.5.5 0 0 1 16 3.5v8.662a1 1 0 0 1-.629.928l-7.185 2.874a.5.5 0 0 1-.372 0L.63 13.09a1 1 0 0 1-.63-.928V3.5a.5.5 0 0 1 .314-.464L7.443.184z"/>
                    </svg>
                    <p class="text-center">Vector visualization requires Three.js.</p>
                    <p class="text-center text-muted small">Add <code>&lt;script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"&gt;&lt;/script&gt;</code> to your page.</p>
                </div>
            `;
            return;
        }
        
        // Create Three.js scene
        this.scene = new THREE.Scene();
        
        // Set background color based on dark mode
        const isDarkMode = document.body.classList.contains('dark-mode');
        this.scene.background = new THREE.Color(isDarkMode ? this.options.darkModeBackgroundColor : this.options.backgroundColor);
        
        // Create camera
        this.camera = new THREE.PerspectiveCamera(75, this.canvasContainer.clientWidth / this.canvasContainer.clientHeight, 0.1, 1000);
        this.camera.position.z = 2;
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.canvasContainer.clientWidth, this.canvasContainer.clientHeight);
        this.canvasContainer.appendChild(this.renderer.domElement);
        
        // Add camera controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.25;
        
        // Add lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(0, 1, 1);
        this.scene.add(directionalLight);
        
        // Add coordinate axes for reference
        const axesHelper = new THREE.AxesHelper(1);
        this.scene.add(axesHelper);
        
        // Point selection functionality
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        // Add event listeners
        window.addEventListener('resize', () => this.onWindowResize());
        this.renderer.domElement.addEventListener('click', (event) => this.onCanvasClick(event));
        
        // Start animation loop
        this.animate();
    }
    
    // Load visualization data
    loadData(data) {
        if (!data || !data.points || !data.points.length) {
            console.error('Invalid data format for visualization');
            return;
        }
        
        // Clear existing points
        this.clearPoints();
        
        // Store data
        this.points = data.points;
        this.metadata = data.metadata || [];
        this.contents = data.contents || [];
        this.similarities = data.similarities || [];
        this.queryPoint = data.query_point;
        
        // Create points visualization
        this.createPointsVisualization();
    }
    
    // Clear all points from visualization
    clearPoints() {
        // Remove existing points from scene
        this.scene.children = this.scene.children.filter(child => !(child instanceof THREE.Points || child instanceof THREE.Sprite));
        
        // Reset state
        this.points = [];
        this.queryPoint = null;
        this.selectedPointIndex = null;
        this.metadata = [];
        this.contents = [];
        this.similarities = [];
        
        // Hide info panel
        this.infoPanel.style.display = 'none';
    }
    
    // Create points visualization
    createPointsVisualization() {
        if (!this.points.length) return;
        
        // Limit number of points to display
        const limitedPoints = this.points.slice(0, this.options.maxPoints);
        
        // Create geometry for result points
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(limitedPoints.length * 3);
        
        // Fill positions
        for (let i = 0; i < limitedPoints.length; i++) {
            const point = limitedPoints[i];
            positions[i * 3] = point[0];     // x
            positions[i * 3 + 1] = point[1]; // y
            positions[i * 3 + 2] = point[2]; // z
        }
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        // Create materials based on color scheme
        const colorScheme = this.getColorScheme();
        const pointsMaterial = new THREE.PointsMaterial({
            color: colorScheme.points,
            size: this.options.pointSize,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.8,
        });
        
        // Create points mesh
        const pointsMesh = new THREE.Points(geometry, pointsMaterial);
        this.scene.add(pointsMesh);
        
        // Add query point if available
        if (this.queryPoint) {
            const queryGeometry = new THREE.BufferGeometry();
            const queryPositions = new Float32Array(3);
            
            queryPositions[0] = this.queryPoint[0];
            queryPositions[1] = this.queryPoint[1];
            queryPositions[2] = this.queryPoint[2];
            
            queryGeometry.setAttribute('position', new THREE.BufferAttribute(queryPositions, 3));
            
            const queryMaterial = new THREE.PointsMaterial({
                color: colorScheme.query,
                size: this.options.queryPointSize,
                sizeAttenuation: true,
            });
            
            const queryPointMesh = new THREE.Points(queryGeometry, queryMaterial);
            this.scene.add(queryPointMesh);
        }
        
        // Add labels if enabled
        if (this.options.showLabels) {
            this.addLabels(limitedPoints);
        }
        
        // Reset camera view
        this.resetView();
    }
    
    // Add text labels to points
    addLabels(points) {
        if (!points.length) return;
        
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        
        for (let i = 0; i < points.length; i++) {
            // Only add labels for points with metadata
            if (!this.metadata[i]) continue;
            
            const point = points[i];
            
            // Create label text
            const labelText = this.metadata[i].title || `Point ${i+1}`;
            
            // Set canvas size
            canvas.width = 256;
            canvas.height = 64;
            context.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw text
            context.font = '24px Arial';
            context.fillStyle = this.options.highContrast ? '#FFFFFF' : '#000000';
            context.textAlign = 'center';
            context.textBaseline = 'middle';
            context.fillText(labelText.substring(0, 20), canvas.width / 2, canvas.height / 2);
            
            // Create texture
            const texture = new THREE.CanvasTexture(canvas);
            texture.needsUpdate = true;
            
            // Create sprite material
            const spriteMaterial = new THREE.SpriteMaterial({
                map: texture,
                transparent: true,
                opacity: 0.8,
            });
            
            // Create sprite
            const sprite = new THREE.Sprite(spriteMaterial);
            sprite.position.set(point[0], point[1] + 0.1, point[2]);
            sprite.scale.set(0.5, 0.125, 1);
            sprite.userData = { pointIndex: i };
            
            // Add sprite to scene
            this.scene.add(sprite);
        }
    }
    
    // Reset camera view
    resetView() {
        if (!this.controls) return;
        
        // Find bounding box of all points
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        
        // Include both result points and query point in bounds calculation
        const allPoints = [...this.points];
        if (this.queryPoint) {
            allPoints.push(this.queryPoint);
        }
        
        for (const point of allPoints) {
            minX = Math.min(minX, point[0]);
            minY = Math.min(minY, point[1]);
            minZ = Math.min(minZ, point[2]);
            
            maxX = Math.max(maxX, point[0]);
            maxY = Math.max(maxY, point[1]);
            maxZ = Math.max(maxZ, point[2]);
        }
        
        // Set camera position to view all points
        const center = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
        const size = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
        
        this.controls.target.set(center[0], center[1], center[2]);
        this.camera.position.set(
            center[0] + size * 1.5,
            center[1] + size * 1.5,
            center[2] + size * 1.5
        );
        
        this.controls.update();
    }
    
    // Toggle label visibility
    toggleLabels() {
        this.options.showLabels = !this.options.showLabels;
        
        // Update labels
        const sprites = this.scene.children.filter(child => child instanceof THREE.Sprite);
        for (const sprite of sprites) {
            sprite.visible = this.options.showLabels;
        }
    }
    
    // Get appropriate color scheme
    getColorScheme() {
        if (this.options.highContrast) {
            return this.colorSchemes.highContrast;
        } else if (this.options.colorblindMode) {
            return this.colorSchemes.colorblind;
        } else {
            return this.colorSchemes.default;
        }
    }
    
    // Update color scheme based on options
    updateColorScheme() {
        const colorScheme = this.getColorScheme();
        
        // Update legend colors
        const queryLegend = this.container.querySelector('.vector-vis-legend-query');
        const pointLegend = this.container.querySelector('.vector-vis-legend-point');
        const selectedLegend = this.container.querySelector('.vector-vis-legend-selected');
        
        if (queryLegend) queryLegend.style.backgroundColor = colorScheme.query;
        if (pointLegend) pointLegend.style.backgroundColor = colorScheme.points;
        if (selectedLegend) selectedLegend.style.backgroundColor = colorScheme.selected;
        
        // Update point colors if points exist
        const pointsMeshes = this.scene?.children.filter(child => child instanceof THREE.Points);
        if (pointsMeshes && pointsMeshes.length > 0) {
            // First point mesh should be the result points
            if (pointsMeshes[0].material) {
                pointsMeshes[0].material.color = new THREE.Color(colorScheme.points);
            }
            
            // Second point mesh (if exists) should be the query point
            if (pointsMeshes.length > 1 && pointsMeshes[1].material) {
                pointsMeshes[1].material.color = new THREE.Color(colorScheme.query);
            }
        }
    }
    
    // Handle window resize
    onWindowResize() {
        if (!this.camera || !this.renderer) return;
        
        this.camera.aspect = this.canvasContainer.clientWidth / this.canvasContainer.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.canvasContainer.clientWidth, this.canvasContainer.clientHeight);
    }
    
    // Handle canvas click
    onCanvasClick(event) {
        if (!this.raycaster || !this.points.length) return;
        
        // Calculate mouse position
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        // Check for intersections
        this.raycaster.setFromCamera(this.mouse, this.camera);
        
        // Find point meshes
        const pointsMeshes = this.scene.children.filter(child => child instanceof THREE.Points);
        if (!pointsMeshes.length) return;
        
        // Get intersections with result points (first mesh)
        const intersects = this.raycaster.intersectObject(pointsMeshes[0]);
        
        if (intersects.length > 0) {
            // Get point index
            const pointIndex = intersects[0].index;
            
            // Show point info
            this.showPointInfo(pointIndex);
        } else {
            // Check for intersections with sprites (labels)
            const sprites = this.scene.children.filter(child => child instanceof THREE.Sprite);
            const spriteIntersects = this.raycaster.intersectObjects(sprites);
            
            if (spriteIntersects.length > 0) {
                const sprite = spriteIntersects[0].object;
                if (sprite.userData && typeof sprite.userData.pointIndex === 'number') {
                    this.showPointInfo(sprite.userData.pointIndex);
                }
            } else {
                // Hide info panel if clicked on empty space
                this.infoPanel.style.display = 'none';
                this.selectedPointIndex = null;
            }
        }
    }
    
    // Show point information
    showPointInfo(pointIndex) {
        if (pointIndex >= this.points.length) return;
        
        // Set selected point
        this.selectedPointIndex = pointIndex;
        
        // Get point data
        const point = this.points[pointIndex];
        const metadata = this.metadata[pointIndex] || {};
        const content = this.contents[pointIndex] || 'No content available';
        const similarity = this.similarities[pointIndex];
        
        // Update info panel
        this.infoTitle.textContent = metadata.title || `Result ${pointIndex + 1}`;
        this.infoContent.textContent = content;
        
        // Show similarity if available
        if (similarity !== undefined) {
            const percent = (similarity * 100).toFixed(2);
            this.infoSimilarity.innerHTML = `<div class="progress" role="progressbar" aria-label="Similarity score" aria-valuenow="${percent}" aria-valuemin="0" aria-valuemax="100">
                <div class="progress-bar bg-success" style="width: ${percent}%">${percent}%</div>
            </div>`;
        } else {
            this.infoSimilarity.innerHTML = '';
        }
        
        // Show info panel
        this.infoPanel.style.display = 'block';
        
        // Update dark mode styling
        if (document.body.classList.contains('dark-mode')) {
            this.infoPanel.style.backgroundColor = 'rgba(40,40,40,0.9)';
            this.infoPanel.style.color = '#f0f0f0';
            this.infoPanel.style.borderTop = '1px solid #444';
        } else {
            this.infoPanel.style.backgroundColor = 'rgba(255,255,255,0.9)';
            this.infoPanel.style.color = '#333';
            this.infoPanel.style.borderTop = '1px solid #ddd';
        }
    }
    
    // Animation loop
    animate() {
        if (!this.renderer || !this.scene || !this.camera) return;
        
        requestAnimationFrame(() => this.animate());
        
        if (this.controls) {
            this.controls.update();
        }
        
        this.renderer.render(this.scene, this.camera);
    }
    
    // Setup accessibility features
    setupAccessibility() {
        // Make visualization container focusable
        this.container.setAttribute('tabindex', '0');
        this.container.setAttribute('role', 'region');
        this.container.setAttribute('aria-label', 'Vector embedding visualization in 3D space');
        
        // Add keyboard navigation
        this.container.addEventListener('keydown', (event) => {
            switch (event.key) {
                case 'r':
                    this.resetView();
                    break;
                case 'l':
                    this.toggleLabels();
                    break;
                case 'ArrowRight':
                    if (this.selectedPointIndex !== null) {
                        const nextIndex = (this.selectedPointIndex + 1) % this.points.length;
                        this.showPointInfo(nextIndex);
                    } else if (this.points.length > 0) {
                        this.showPointInfo(0);
                    }
                    break;
                case 'ArrowLeft':
                    if (this.selectedPointIndex !== null) {
                        const prevIndex = (this.selectedPointIndex - 1 + this.points.length) % this.points.length;
                        this.showPointInfo(prevIndex);
                    } else if (this.points.length > 0) {
                        this.showPointInfo(this.points.length - 1);
                    }
                    break;
                case 'Escape':
                    this.infoPanel.style.display = 'none';
                    this.selectedPointIndex = null;
                    break;
            }
        });
        
        // Add screen reader instructions
        const instructions = document.createElement('div');
        instructions.className = 'visually-hidden';
        instructions.textContent = 'Use arrow keys to navigate between points. Press R to reset view, L to toggle labels, and Escape to close details.';
        this.container.appendChild(instructions);
    }
    
    // Set options
    setOptions(options) {
        this.options = Object.assign(this.options, options);
        
        // Update visualization based on new options
        this.updateColorScheme();
        
        // Update scene background if dark mode changed
        if (this.scene) {
            const isDarkMode = document.body.classList.contains('dark-mode');
            this.scene.background = new THREE.Color(isDarkMode ? this.options.darkModeBackgroundColor : this.options.backgroundColor);
        }
        
        // Redraw points if needed
        if (this.points.length > 0) {
            this.createPointsVisualization();
        }
    }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VectorVisualization;
}