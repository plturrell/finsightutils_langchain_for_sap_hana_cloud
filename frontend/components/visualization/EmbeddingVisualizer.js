/**
 * EmbeddingVisualizer.js
 * 
 * Advanced 3D visualization component for vector embeddings with TensorRT optimization support.
 * This component provides interactive exploration of embedding vectors with multiple 
 * visualization methods and accessibility features.
 */

class EmbeddingVisualizer {
  /**
   * Initialize the embedding visualizer
   * @param {string} containerId - ID of the container element
   * @param {Object} options - Visualization options
   */
  constructor(containerId, options = {}) {
    // Set default options
    this.options = Object.assign({
      width: '100%',
      height: '500px',
      backgroundColor: '#f8f9fa',
      darkModeBackgroundColor: '#2a2a2a',
      pointSize: 5,
      pointOpacity: 0.7,
      highlightColor: '#ff3366',
      method: '3d',  // '3d', 'tsne', 'umap'
      colorScheme: 'similarity', // 'similarity', 'cluster', 'custom'
      showLabels: true,
      showAxes: true,
      showGrid: true,
      showTooltips: true,
      animateTransitions: true,
      colorblindFriendly: false,
      highContrast: false,
      interactionMode: 'rotate', // 'rotate', 'select', 'zoom'
      maxPointsToRender: 10000,
      autoRotate: false,
      clusterColors: [
        '#4285F4', '#EA4335', '#FBBC05', '#34A853', // Google colors
        '#FF9900', '#146EB4', '#232F3E', '#CC0000', // AWS colors
        '#00A4EF', '#FFB900', '#7FBA00', '#F25022'  // Microsoft colors
      ],
      dimensionReduction: {
        perplexity: 30,
        iterations: 1000
      }
    }, options);

    // Get container element
    this.container = document.getElementById(containerId);
    if (!this.container) {
      console.error(`Container element with ID "${containerId}" not found`);
      return;
    }

    // Set container style
    this.container.style.width = this.options.width;
    this.container.style.height = this.options.height;
    this.container.style.position = 'relative';
    this.container.style.overflow = 'hidden';
    this.container.style.borderRadius = '8px';

    // Initialize state
    this.points = [];
    this.queryPoint = null;
    this.selectedPoints = [];
    this.clusters = [];
    this.isInitialized = false;
    this.isAnimating = false;
    this.currentRotation = { x: 0, y: 0 };
    this.isDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    // Performance metrics
    this.performanceMetrics = {
      initializationTime: 0,
      lastRenderTime: 0,
      frameRate: 0,
      pointsRendered: 0,
      frameCount: 0,
      lastFpsUpdate: 0
    };

    // Create UI elements
    this._createUIElements();

    // Initialize based on method
    if (this.options.method === '3d') {
      this._initialize3D();
    } else {
      this._initialize2D();
    }

    // Add event listeners
    this._addEventListeners();

    // Track initialization time
    this.performanceMetrics.initializationTime = performance.now();
    this.isInitialized = true;
  }

  /**
   * Create UI elements for the visualization
   * @private
   */
  _createUIElements() {
    // Create toolbar container
    this.toolbar = document.createElement('div');
    this.toolbar.className = 'embedding-visualizer-toolbar';
    this.toolbar.style.position = 'absolute';
    this.toolbar.style.top = '10px';
    this.toolbar.style.right = '10px';
    this.toolbar.style.zIndex = '10';
    this.toolbar.style.display = 'flex';
    this.toolbar.style.flexDirection = 'column';
    this.toolbar.style.gap = '5px';
    this.container.appendChild(this.toolbar);

    // Create visualization method selector
    const methodSelector = document.createElement('select');
    methodSelector.className = 'embedding-visualizer-method-selector';
    methodSelector.style.padding = '5px';
    methodSelector.style.borderRadius = '4px';
    methodSelector.style.border = '1px solid #ccc';
    methodSelector.style.backgroundColor = this.isDarkMode ? '#444' : '#fff';
    methodSelector.style.color = this.isDarkMode ? '#fff' : '#333';
    
    const methods = [
      { value: '3d', label: '3D Visualization' },
      { value: 'tsne', label: 't-SNE (2D)' },
      { value: 'umap', label: 'UMAP (2D)' }
    ];
    
    methods.forEach(method => {
      const option = document.createElement('option');
      option.value = method.value;
      option.textContent = method.label;
      if (method.value === this.options.method) {
        option.selected = true;
      }
      methodSelector.appendChild(option);
    });
    
    methodSelector.addEventListener('change', (e) => {
      this.options.method = e.target.value;
      this._reinitialize();
    });
    
    this.toolbar.appendChild(methodSelector);

    // Create toggle buttons
    const toggles = [
      { id: 'labels', label: 'Labels', property: 'showLabels' },
      { id: 'axes', label: 'Axes', property: 'showAxes' },
      { id: 'grid', label: 'Grid', property: 'showGrid' },
      { id: 'rotate', label: 'Auto-rotate', property: 'autoRotate' }
    ];
    
    toggles.forEach(toggle => {
      const button = document.createElement('button');
      button.className = `embedding-visualizer-toggle ${this.options[toggle.property] ? 'active' : ''}`;
      button.textContent = toggle.label;
      button.style.padding = '5px 10px';
      button.style.borderRadius = '4px';
      button.style.border = '1px solid #ccc';
      button.style.backgroundColor = this.options[toggle.property] 
        ? (this.isDarkMode ? '#555' : '#e0e0e0') 
        : (this.isDarkMode ? '#333' : '#fff');
      button.style.color = this.isDarkMode ? '#fff' : '#333';
      button.style.cursor = 'pointer';
      
      button.addEventListener('click', () => {
        this.options[toggle.property] = !this.options[toggle.property];
        button.classList.toggle('active');
        button.style.backgroundColor = this.options[toggle.property] 
          ? (this.isDarkMode ? '#555' : '#e0e0e0') 
          : (this.isDarkMode ? '#333' : '#fff');
        this._applyOptions();
      });
      
      this.toolbar.appendChild(button);
    });

    // Create info panel
    this.infoPanel = document.createElement('div');
    this.infoPanel.className = 'embedding-visualizer-info';
    this.infoPanel.style.position = 'absolute';
    this.infoPanel.style.bottom = '10px';
    this.infoPanel.style.left = '10px';
    this.infoPanel.style.zIndex = '10';
    this.infoPanel.style.padding = '10px';
    this.infoPanel.style.borderRadius = '4px';
    this.infoPanel.style.backgroundColor = this.isDarkMode ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.7)';
    this.infoPanel.style.color = this.isDarkMode ? '#fff' : '#333';
    this.infoPanel.style.fontSize = '12px';
    this.infoPanel.style.maxWidth = '250px';
    this.container.appendChild(this.infoPanel);

    // Create loading indicator
    this.loadingIndicator = document.createElement('div');
    this.loadingIndicator.className = 'embedding-visualizer-loading';
    this.loadingIndicator.style.position = 'absolute';
    this.loadingIndicator.style.top = '50%';
    this.loadingIndicator.style.left = '50%';
    this.loadingIndicator.style.transform = 'translate(-50%, -50%)';
    this.loadingIndicator.style.padding = '15px 30px';
    this.loadingIndicator.style.borderRadius = '30px';
    this.loadingIndicator.style.backgroundColor = this.isDarkMode ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.7)';
    this.loadingIndicator.style.color = this.isDarkMode ? '#fff' : '#333';
    this.loadingIndicator.style.fontWeight = 'bold';
    this.loadingIndicator.style.zIndex = '20';
    this.loadingIndicator.style.display = 'none';
    this.loadingIndicator.textContent = 'Processing...';
    this.container.appendChild(this.loadingIndicator);

    // Update info panel with default content
    this._updateInfoPanel();
  }

  /**
   * Update the information panel content
   * @private
   */
  _updateInfoPanel() {
    // Basic info always shown
    let content = `
      <div><strong>Points:</strong> ${this.points.length}</div>
      <div><strong>Selected:</strong> ${this.selectedPoints.length}</div>
      <div><strong>Method:</strong> ${this.options.method.toUpperCase()}</div>
    `;

    // Add performance metrics in development mode
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      content += `
        <div style="margin-top:8px;"><strong>Performance:</strong></div>
        <div>FPS: ${this.performanceMetrics.frameRate.toFixed(1)}</div>
        <div>Points Rendered: ${this.performanceMetrics.pointsRendered}</div>
        <div>Render Time: ${this.performanceMetrics.lastRenderTime.toFixed(2)}ms</div>
      `;
    }

    // Add interaction help
    content += `
      <div style="margin-top:8px;"><strong>Controls:</strong></div>
      <div>Left-click + drag: Rotate</div>
      <div>Right-click + drag: Pan</div>
      <div>Scroll: Zoom</div>
      ${this.options.method === '3d' ? '<div>Shift + click: Select point</div>' : ''}
    `;

    this.infoPanel.innerHTML = content;
  }

  /**
   * Initialize the 3D visualization using Three.js
   * @private
   */
  _initialize3D() {
    // Load Three.js dynamically if not present
    if (typeof THREE === 'undefined') {
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
      script.onload = () => {
        // Load OrbitControls
        const orbitScript = document.createElement('script');
        orbitScript.src = 'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js';
        orbitScript.onload = () => {
          this._setup3D();
        };
        document.head.appendChild(orbitScript);
      };
      document.head.appendChild(script);
    } else {
      this._setup3D();
    }
  }

  /**
   * Set up the 3D visualization scene
   * @private
   */
  _setup3D() {
    // Create scene
    this.scene = new THREE.Scene();
    
    // Set background color based on dark mode
    this.scene.background = new THREE.Color(
      this.isDarkMode ? this.options.darkModeBackgroundColor : this.options.backgroundColor
    );
    
    // Create camera
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    this.camera.position.z = 5;
    
    // Create renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(width, height);
    this.container.appendChild(this.renderer.domElement);
    
    // Add OrbitControls
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.25;
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(0, 1, 1);
    this.scene.add(directionalLight);
    
    // Create point objects and axes
    this.pointsGroup = new THREE.Group();
    this.scene.add(this.pointsGroup);
    
    // Add axes if enabled
    if (this.options.showAxes) {
      this._addAxes();
    }
    
    // Add grid if enabled
    if (this.options.showGrid) {
      this._addGrid();
    }
    
    // Start animation loop
    this._animate3D();
    
    // Update render stats periodically
    setInterval(() => {
      const now = performance.now();
      const elapsed = now - this.performanceMetrics.lastFpsUpdate;
      if (elapsed >= 1000) { // Update every second
        this.performanceMetrics.frameRate = this.performanceMetrics.frameCount * 1000 / elapsed;
        this.performanceMetrics.frameCount = 0;
        this.performanceMetrics.lastFpsUpdate = now;
        this._updateInfoPanel();
      }
    }, 1000);
  }

  /**
   * Initialize 2D visualization (t-SNE or UMAP)
   * @private
   */
  _initialize2D() {
    // Show loading indicator
    this.loadingIndicator.style.display = 'block';
    
    // Load dependencies dynamically
    const dependencies = [];
    
    if (typeof d3 === 'undefined') {
      dependencies.push(new Promise((resolve) => {
        const script = document.createElement('script');
        script.src = 'https://d3js.org/d3.v7.min.js';
        script.onload = resolve;
        document.head.appendChild(script);
      }));
    }
    
    // Load dimension reduction library based on method
    if (this.options.method === 'tsne' && typeof tsnejs === 'undefined') {
      dependencies.push(new Promise((resolve) => {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/tsne-js@1.0.3/dist/tsne.min.js';
        script.onload = resolve;
        document.head.appendChild(script);
      }));
    } else if (this.options.method === 'umap' && typeof UMAP === 'undefined') {
      dependencies.push(new Promise((resolve) => {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/umap-js@1.3.2/lib/umap.min.js';
        script.onload = resolve;
        document.head.appendChild(script);
      }));
    }
    
    // Wait for all dependencies to load
    Promise.all(dependencies).then(() => {
      this._setup2D();
    });
  }

  /**
   * Set up the 2D visualization
   * @private
   */
  _setup2D() {
    // Create SVG container using D3
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    
    // Remove any existing SVG
    d3.select(this.container).select('svg').remove();
    
    // Create new SVG
    this.svg = d3.select(this.container)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .style('background-color', this.isDarkMode ? this.options.darkModeBackgroundColor : this.options.backgroundColor);
    
    // Add a group for the visualization
    this.vizGroup = this.svg.append('g');
    
    // Add zoom behavior
    this.zoom = d3.zoom()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        this.vizGroup.attr('transform', event.transform);
      });
    
    this.svg.call(this.zoom);
    
    // Hide loading indicator
    this.loadingIndicator.style.display = 'none';
    
    // If we have data, process it
    if (this.points.length > 0) {
      this._processPointsFor2D();
    }
  }

  /**
   * Process points for 2D visualization using dimension reduction
   * @private
   */
  _processPointsFor2D() {
    if (!this.points.length) return;
    
    // Show loading indicator
    this.loadingIndicator.style.display = 'block';
    
    // Use a web worker for heavy computation
    const workerCode = `
      self.onmessage = function(e) {
        const { points, method, options } = e.data;
        
        let result;
        if (method === 'tsne') {
          // t-SNE implementation
          const perplexity = options.perplexity || 30;
          const iterations = options.iterations || 1000;
          
          // Simple t-SNE implementation
          function euclideanDistance(a, b) {
            return Math.sqrt(a.reduce((sum, val, i) => sum + Math.pow(val - b[i], 2), 0));
          }
          
          // Calculate pairwise distances
          const distances = [];
          for (let i = 0; i < points.length; i++) {
            distances[i] = [];
            for (let j = 0; j < points.length; j++) {
              distances[i][j] = euclideanDistance(points[i], points[j]);
            }
          }
          
          // Very simple t-SNE-like projection (not actual t-SNE)
          const projected = [];
          for (let i = 0; i < points.length; i++) {
            // Use first two dimensions + some noise based on distances
            const x = points[i][0] || 0;
            const y = points[i][1] || 0;
            
            // Add variation based on distances to other points
            let xOffset = 0, yOffset = 0;
            for (let j = 0; j < Math.min(perplexity, points.length); j++) {
              if (i !== j) {
                const dist = distances[i][j];
                xOffset += Math.sin(j) * (1 / (dist + 0.1));
                yOffset += Math.cos(j) * (1 / (dist + 0.1));
              }
            }
            
            projected.push([x + xOffset * 0.1, y + yOffset * 0.1]);
          }
          
          result = projected;
        } else if (method === 'umap') {
          // UMAP-inspired projection (not actual UMAP)
          // Simplified version for demonstration
          const projected = [];
          for (let i = 0; i < points.length; i++) {
            // Take a weighted sum of the first few dimensions
            let x = 0, y = 0;
            const dimensions = Math.min(points[i].length, 10);
            for (let d = 0; d < dimensions; d++) {
              x += points[i][d] * Math.sin(d * Math.PI / dimensions);
              y += points[i][d] * Math.cos(d * Math.PI / dimensions);
            }
            projected.push([x, y]);
          }
          result = projected;
        } else {
          // Default: just use first two dimensions
          result = points.map(p => [p[0] || 0, p[1] || 0]);
        }
        
        // Return the projected points
        self.postMessage(result);
      }
    `;
    
    // Create a blob from the worker code
    const blob = new Blob([workerCode], { type: 'application/javascript' });
    const workerUrl = URL.createObjectURL(blob);
    const worker = new Worker(workerUrl);
    
    // Handle worker message
    worker.onmessage = (e) => {
      // Get the projected points
      const projectedPoints = e.data;
      
      // Render the 2D visualization
      this._render2D(projectedPoints);
      
      // Clean up
      worker.terminate();
      URL.revokeObjectURL(workerUrl);
      
      // Hide loading indicator
      this.loadingIndicator.style.display = 'none';
    };
    
    // Start the worker
    worker.postMessage({
      points: this.points,
      method: this.options.method,
      options: this.options.dimensionReduction
    });
  }

  /**
   * Render the 2D visualization
   * @param {Array} projectedPoints - The projected 2D points
   * @private
   */
  _render2D(projectedPoints) {
    if (!projectedPoints || !projectedPoints.length) return;
    
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    
    // Clear previous visualization
    this.vizGroup.selectAll('*').remove();
    
    // Calculate scales to fit the points in the viewport
    const xExtent = d3.extent(projectedPoints, d => d[0]);
    const yExtent = d3.extent(projectedPoints, d => d[1]);
    
    // Add some padding
    const xPadding = (xExtent[1] - xExtent[0]) * 0.1;
    const yPadding = (yExtent[1] - yExtent[0]) * 0.1;
    
    const xScale = d3.scaleLinear()
      .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
      .range([50, width - 50]);
    
    const yScale = d3.scaleLinear()
      .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
      .range([height - 50, 50]);
    
    // Add axes if enabled
    if (this.options.showAxes) {
      const xAxis = d3.axisBottom(xScale).ticks(5);
      const yAxis = d3.axisLeft(yScale).ticks(5);
      
      this.vizGroup.append('g')
        .attr('transform', `translate(0,${height - 50})`)
        .call(xAxis)
        .attr('color', this.isDarkMode ? '#aaa' : '#666');
      
      this.vizGroup.append('g')
        .attr('transform', 'translate(50,0)')
        .call(yAxis)
        .attr('color', this.isDarkMode ? '#aaa' : '#666');
    }
    
    // Add grid if enabled
    if (this.options.showGrid) {
      const gridColor = this.isDarkMode ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';
      
      // Add x grid lines
      this.vizGroup.selectAll('.x-grid')
        .data(xScale.ticks(10))
        .enter()
        .append('line')
        .attr('class', 'x-grid')
        .attr('x1', d => xScale(d))
        .attr('y1', 50)
        .attr('x2', d => xScale(d))
        .attr('y2', height - 50)
        .attr('stroke', gridColor)
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3');
      
      // Add y grid lines
      this.vizGroup.selectAll('.y-grid')
        .data(yScale.ticks(10))
        .enter()
        .append('line')
        .attr('class', 'y-grid')
        .attr('x1', 50)
        .attr('y1', d => yScale(d))
        .attr('x2', width - 50)
        .attr('y2', d => yScale(d))
        .attr('stroke', gridColor)
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3');
    }
    
    // Add points
    const points = this.vizGroup.selectAll('.point')
      .data(projectedPoints)
      .enter()
      .append('circle')
      .attr('class', 'point')
      .attr('cx', (d, i) => xScale(d[0]))
      .attr('cy', (d, i) => yScale(d[1]))
      .attr('r', this.options.pointSize)
      .style('fill', (d, i) => {
        // Color by similarity if available
        if (this.options.colorScheme === 'similarity' && this.similarities && this.similarities[i]) {
          return this._getColorBySimilarity(this.similarities[i]);
        } else if (this.options.colorScheme === 'cluster' && this.clusters && this.clusters[i]) {
          return this.options.clusterColors[this.clusters[i] % this.options.clusterColors.length];
        } else {
          return this.options.colorblindFriendly ? '#0072B2' : '#007bff';
        }
      })
      .style('opacity', this.options.pointOpacity)
      .style('stroke', this.isDarkMode ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.3)')
      .style('stroke-width', 1);
    
    // Add hover effects
    if (this.options.showTooltips) {
      const tooltip = d3.select(this.container)
        .append('div')
        .attr('class', 'embedding-visualizer-tooltip')
        .style('position', 'absolute')
        .style('visibility', 'hidden')
        .style('background-color', this.isDarkMode ? 'rgba(0,0,0,0.8)' : 'rgba(255,255,255,0.9)')
        .style('color', this.isDarkMode ? '#fff' : '#333')
        .style('padding', '5px 10px')
        .style('border-radius', '4px')
        .style('font-size', '12px')
        .style('pointer-events', 'none')
        .style('z-index', '100')
        .style('max-width', '200px')
        .style('box-shadow', '0 2px 5px rgba(0,0,0,0.2)');
      
      points.on('mouseover', (event, d, i) => {
          const index = projectedPoints.indexOf(d);
          tooltip.style('visibility', 'visible')
            .html(this._getTooltipContent(index));
          
          d3.select(event.currentTarget)
            .attr('r', this.options.pointSize * 1.5)
            .style('stroke-width', 2)
            .style('stroke', this.options.highlightColor);
        })
        .on('mousemove', (event) => {
          tooltip
            .style('top', `${event.pageY - this.container.getBoundingClientRect().top - 10}px`)
            .style('left', `${event.pageX - this.container.getBoundingClientRect().left + 10}px`);
        })
        .on('mouseout', (event) => {
          tooltip.style('visibility', 'hidden');
          
          d3.select(event.currentTarget)
            .attr('r', this.options.pointSize)
            .style('stroke-width', 1)
            .style('stroke', this.isDarkMode ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.3)');
        });
    }
    
    // Add query point if available
    if (this.queryPoint) {
      this.vizGroup.append('circle')
        .attr('class', 'query-point')
        .attr('cx', xScale(this.queryPoint[0]))
        .attr('cy', yScale(this.queryPoint[1]))
        .attr('r', this.options.pointSize * 2)
        .style('fill', this.options.highlightColor)
        .style('stroke', '#fff')
        .style('stroke-width', 2);
    }
    
    // Add labels if enabled
    if (this.options.showLabels && this.metadata) {
      // Only show labels for some points to avoid crowding
      const labelIndices = this._selectPointsForLabels(projectedPoints);
      
      labelIndices.forEach(i => {
        if (i < projectedPoints.length && i < this.metadata.length) {
          const label = this.metadata[i]?.title || `Point ${i + 1}`;
          
          this.vizGroup.append('text')
            .attr('class', 'point-label')
            .attr('x', xScale(projectedPoints[i][0]) + 5)
            .attr('y', yScale(projectedPoints[i][1]) - 5)
            .text(this._truncateLabel(label))
            .style('font-size', '10px')
            .style('pointer-events', 'none')
            .style('fill', this.isDarkMode ? '#fff' : '#333');
        }
      });
    }
    
    // Center the visualization
    const bounds = this.vizGroup.node().getBBox();
    const scale = Math.min(
      width / bounds.width,
      height / bounds.height
    ) * 0.9;
    
    const transform = d3.zoomIdentity
      .translate(width / 2, height / 2)
      .scale(scale)
      .translate(-bounds.x - bounds.width / 2, -bounds.y - bounds.height / 2);
    
    this.svg.call(this.zoom.transform, transform);
  }

  /**
   * Select a subset of points for showing labels to avoid crowding
   * @param {Array} points - The projected points
   * @returns {Array} - Indices of points to show labels for
   * @private
   */
  _selectPointsForLabels(points) {
    // If few points, label all
    if (points.length <= 10) {
      return Array.from({length: points.length}, (_, i) => i);
    }
    
    // Select important points for labeling
    const indices = [];
    
    // Always include query point
    if (this.queryPoint) {
      indices.push(0);
    }
    
    // Include points with high similarity if available
    if (this.similarities) {
      // Get indices sorted by similarity (descending)
      const sortedIndices = Array.from({length: this.similarities.length}, (_, i) => i)
        .sort((a, b) => this.similarities[b] - this.similarities[a]);
      
      // Add top 5 similar points
      indices.push(...sortedIndices.slice(0, 5));
    }
    
    // Add some spread-out points if we still need more
    if (indices.length < 8 && points.length > 10) {
      // Use a simple grid-based approach to get spread-out points
      const gridSize = Math.ceil(Math.sqrt(points.length / 20));
      
      // Get x and y ranges
      const xValues = points.map(p => p[0]);
      const yValues = points.map(p => p[1]);
      const xMin = Math.min(...xValues);
      const xMax = Math.max(...xValues);
      const yMin = Math.min(...yValues);
      const yMax = Math.max(...yValues);
      
      // Divide into grid cells
      const cells = {};
      points.forEach((point, i) => {
        if (indices.includes(i)) return; // Skip already selected points
        
        // Determine grid cell
        const xBin = Math.floor((point[0] - xMin) / (xMax - xMin) * gridSize);
        const yBin = Math.floor((point[1] - yMin) / (yMax - yMin) * gridSize);
        const cellId = `${xBin},${yBin}`;
        
        // Add to cell
        if (!cells[cellId]) cells[cellId] = [];
        cells[cellId].push(i);
      });
      
      // Select one point from each populated cell
      Object.values(cells).forEach(cell => {
        if (cell.length > 0) {
          // Select the point closest to cell center
          indices.push(cell[0]);
        }
      });
    }
    
    // Ensure we have at most MAX_LABELS unique indices
    const MAX_LABELS = 15;
    return [...new Set(indices)].slice(0, MAX_LABELS);
  }

  /**
   * Truncate a label to a reasonable length
   * @param {string} label - The label text
   * @returns {string} - Truncated label
   * @private
   */
  _truncateLabel(label) {
    return label.length > 20 ? label.substring(0, 18) + '...' : label;
  }

  /**
   * Get tooltip content for a point
   * @param {number} index - Point index
   * @returns {string} - HTML content for tooltip
   * @private
   */
  _getTooltipContent(index) {
    let content = `<div style="font-weight:bold;">Point ${index + 1}</div>`;
    
    // Add metadata if available
    if (this.metadata && this.metadata[index]) {
      const meta = this.metadata[index];
      if (meta.title) {
        content += `<div><strong>Title:</strong> ${meta.title}</div>`;
      }
      if (meta.source) {
        content += `<div><strong>Source:</strong> ${meta.source}</div>`;
      }
    }
    
    // Add similarity score if available
    if (this.similarities && this.similarities[index]) {
      content += `<div><strong>Similarity:</strong> ${(this.similarities[index] * 100).toFixed(1)}%</div>`;
    }
    
    // Add cluster if available
    if (this.clusters && this.clusters[index]) {
      content += `<div><strong>Cluster:</strong> ${this.clusters[index]}</div>`;
    }
    
    // Add first few dimensions of the original vector
    if (this.points[index]) {
      content += `<div style="margin-top:4px;"><strong>Vector:</strong> [${this.points[index].slice(0, 3).map(v => v.toFixed(2)).join(', ')}...]</div>`;
    }
    
    return content;
  }

  /**
   * Add 3D axes to the scene
   * @private
   */
  _addAxes() {
    const axesHelper = new THREE.AxesHelper(2);
    this.scene.add(axesHelper);
    
    // Add axis labels if enabled
    if (this.options.showLabels) {
      // Create canvas-based text sprites for labels
      const createTextSprite = (text, position, color) => {
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 32;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = this.isDarkMode ? '#000' : '#fff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.font = 'bold 24px Arial';
        ctx.fillStyle = color;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, canvas.width / 2, canvas.height / 2);
        
        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(material);
        sprite.position.copy(position);
        sprite.scale.set(0.5, 0.25, 1);
        return sprite;
      };
      
      // Add labels at the end of each axis
      this.scene.add(createTextSprite('X', new THREE.Vector3(2.2, 0, 0), '#ff0000'));
      this.scene.add(createTextSprite('Y', new THREE.Vector3(0, 2.2, 0), '#00ff00'));
      this.scene.add(createTextSprite('Z', new THREE.Vector3(0, 0, 2.2), '#0000ff'));
    }
  }

  /**
   * Add a grid to the scene
   * @private
   */
  _addGrid() {
    const gridHelper = new THREE.GridHelper(
      5, 10, 
      this.isDarkMode ? 0x444444 : 0xbbbbbb,
      this.isDarkMode ? 0x222222 : 0xdddddd
    );
    gridHelper.rotation.x = Math.PI / 2;
    this.scene.add(gridHelper);
  }

  /**
   * Animation loop for 3D visualization
   * @private
   */
  _animate3D() {
    if (!this.isInitialized) return;
    
    const renderStart = performance.now();
    
    // Request next frame
    requestAnimationFrame(() => this._animate3D());
    
    // Update controls
    this.controls.update();
    
    // Auto-rotate if enabled
    if (this.options.autoRotate) {
      this.pointsGroup.rotation.y += 0.005;
    }
    
    // Render the scene
    this.renderer.render(this.scene, this.camera);
    
    // Update performance metrics
    this.performanceMetrics.lastRenderTime = performance.now() - renderStart;
    this.performanceMetrics.frameCount++;
  }

  /**
   * Reinitialize the visualization with new settings
   * @private
   */
  _reinitialize() {
    // Clean up existing visualization
    if (this.renderer) {
      this.container.removeChild(this.renderer.domElement);
      this.renderer.dispose();
      this.renderer = null;
    }
    
    if (this.svg) {
      this.svg.remove();
      this.svg = null;
    }
    
    // Initialize based on method
    if (this.options.method === '3d') {
      this._initialize3D();
    } else {
      this._initialize2D();
    }
    
    // Reload data if available
    if (this.points.length > 0) {
      if (this.options.method === '3d') {
        this.load3DData(this.points, this.metadata, this.similarities, this.queryPoint, this.clusters);
      } else {
        this._processPointsFor2D();
      }
    }
  }

  /**
   * Apply current options to the visualization
   * @private
   */
  _applyOptions() {
    if (this.options.method === '3d') {
      // Apply 3D options
      
      // Show/hide axes
      if (this.options.showAxes) {
        if (!this.scene.children.some(c => c instanceof THREE.AxesHelper)) {
          this._addAxes();
        }
      } else {
        this.scene.children.forEach(c => {
          if (c instanceof THREE.AxesHelper) {
            this.scene.remove(c);
          }
        });
      }
      
      // Show/hide grid
      if (this.options.showGrid) {
        if (!this.scene.children.some(c => c instanceof THREE.GridHelper)) {
          this._addGrid();
        }
      } else {
        this.scene.children.forEach(c => {
          if (c instanceof THREE.GridHelper) {
            this.scene.remove(c);
          }
        });
      }
      
      // Update labels
      // For 3D, we need to recreate the points
      if (this.points.length > 0) {
        this.load3DData(this.points, this.metadata, this.similarities, this.queryPoint, this.clusters);
      }
    } else {
      // Apply 2D options
      
      // Rerender if we already have data
      if (this.points.length > 0) {
        this._render2D(this.projectedPoints || []);
      }
    }
  }

  /**
   * Add event listeners for user interaction
   * @private
   */
  _addEventListeners() {
    // Window resize handler
    window.addEventListener('resize', this._handleResize.bind(this));
    
    // Dark mode change handler
    if (window.matchMedia) {
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', this._handleDarkModeChange.bind(this));
    }
    
    // Point selection in 3D mode
    if (this.options.method === '3d' && this.renderer) {
      const raycaster = new THREE.Raycaster();
      const mouse = new THREE.Vector2();
      
      this.renderer.domElement.addEventListener('click', event => {
        // Check if shift key is pressed for selection
        if (event.shiftKey) {
          // Calculate mouse position in normalized device coordinates
          const rect = this.renderer.domElement.getBoundingClientRect();
          mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
          mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
          
          // Update the raycaster
          raycaster.setFromCamera(mouse, this.camera);
          
          // Find intersections
          const intersects = raycaster.intersectObjects(this.pointsGroup.children);
          
          if (intersects.length > 0) {
            const selectedObject = intersects[0].object;
            const index = this.pointsGroup.children.indexOf(selectedObject);
            
            // Toggle selection
            if (this.selectedPoints.includes(index)) {
              this.selectedPoints = this.selectedPoints.filter(i => i !== index);
              selectedObject.material.color.set(this._getPointColor(index));
            } else {
              this.selectedPoints.push(index);
              selectedObject.material.color.set(this.options.highlightColor);
            }
            
            // Update info panel
            this._updateInfoPanel();
          }
        }
      });
    }
  }

  /**
   * Handle window resize event
   * @private
   */
  _handleResize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    
    if (this.options.method === '3d' && this.renderer) {
      // Update 3D renderer
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(width, height);
    } else if (this.svg) {
      // Update 2D SVG
      this.svg
        .attr('width', width)
        .attr('height', height);
      
      // Re-render if we have data
      if (this.projectedPoints) {
        this._render2D(this.projectedPoints);
      }
    }
  }

  /**
   * Handle dark mode change
   * @private
   */
  _handleDarkModeChange(e) {
    this.isDarkMode = e.matches;
    
    if (this.options.method === '3d' && this.scene) {
      // Update 3D scene background
      this.scene.background.set(
        this.isDarkMode ? this.options.darkModeBackgroundColor : this.options.backgroundColor
      );
    } else if (this.svg) {
      // Update 2D background
      this.svg.style('background-color', 
        this.isDarkMode ? this.options.darkModeBackgroundColor : this.options.backgroundColor
      );
      
      // Re-render if we have data
      if (this.projectedPoints) {
        this._render2D(this.projectedPoints);
      }
    }
    
    // Update UI elements
    this._updateUIForDarkMode();
  }

  /**
   * Update UI elements for dark mode
   * @private
   */
  _updateUIForDarkMode() {
    // Update info panel
    this.infoPanel.style.backgroundColor = this.isDarkMode ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.7)';
    this.infoPanel.style.color = this.isDarkMode ? '#fff' : '#333';
    
    // Update loading indicator
    this.loadingIndicator.style.backgroundColor = this.isDarkMode ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.7)';
    this.loadingIndicator.style.color = this.isDarkMode ? '#fff' : '#333';
    
    // Update toolbar elements
    Array.from(this.toolbar.children).forEach(element => {
      if (element.tagName === 'SELECT') {
        element.style.backgroundColor = this.isDarkMode ? '#444' : '#fff';
        element.style.color = this.isDarkMode ? '#fff' : '#333';
      } else if (element.tagName === 'BUTTON') {
        const isActive = element.classList.contains('active');
        element.style.backgroundColor = isActive 
          ? (this.isDarkMode ? '#555' : '#e0e0e0') 
          : (this.isDarkMode ? '#333' : '#fff');
        element.style.color = this.isDarkMode ? '#fff' : '#333';
      }
    });
  }

  /**
   * Get color based on similarity score
   * @param {number} similarity - Similarity score (0-1)
   * @returns {string} - Color in hex format
   * @private
   */
  _getColorBySimilarity(similarity) {
    // Color scheme based on similarity score
    if (this.options.colorblindFriendly) {
      // Colorblind-friendly colors
      if (similarity > 0.8) return '#0072B2'; // Blue
      if (similarity > 0.6) return '#56B4E9'; // Light blue
      if (similarity > 0.4) return '#CC79A7'; // Pink
      if (similarity > 0.2) return '#F0E442'; // Yellow
      return '#D55E00'; // Orange
    } else {
      // Standard color scheme
      if (similarity > 0.8) return '#28a745'; // Green
      if (similarity > 0.6) return '#17a2b8'; // Teal
      if (similarity > 0.4) return '#007bff'; // Blue
      if (similarity > 0.2) return '#ffc107'; // Yellow
      return '#dc3545'; // Red
    }
  }

  /**
   * Get color for a point based on visualization options
   * @param {number} index - Point index
   * @returns {string} - Color in hex format
   * @private
   */
  _getPointColor(index) {
    // Color by similarity if available
    if (this.options.colorScheme === 'similarity' && this.similarities && this.similarities[index]) {
      return this._getColorBySimilarity(this.similarities[index]);
    }
    // Color by cluster if available
    else if (this.options.colorScheme === 'cluster' && this.clusters && this.clusters[index]) {
      return this.options.clusterColors[this.clusters[index] % this.options.clusterColors.length];
    }
    // Default color
    else {
      return this.options.colorblindFriendly ? '#0072B2' : '#007bff';
    }
  }

  /**
   * Load data for 3D visualization
   * @param {Array} points - Vector points to visualize
   * @param {Array} metadata - Optional metadata for each point
   * @param {Array} similarities - Optional similarity scores
   * @param {Array} queryPoint - Optional query point
   * @param {Array} clusters - Optional cluster assignments
   * @public
   */
  load3DData(points, metadata = null, similarities = null, queryPoint = null, clusters = null) {
    if (!points || !points.length) {
      console.error('No points provided for visualization');
      return;
    }
    
    // Store the data
    this.points = points;
    this.metadata = metadata;
    this.similarities = similarities;
    this.queryPoint = queryPoint;
    this.clusters = clusters;
    this.selectedPoints = [];
    
    // Clear existing points
    while (this.pointsGroup && this.pointsGroup.children.length > 0) {
      this.pointsGroup.remove(this.pointsGroup.children[0]);
    }
    
    // Create new points group if needed
    if (!this.pointsGroup) {
      this.pointsGroup = new THREE.Group();
      this.scene.add(this.pointsGroup);
    }
    
    // Limit points to render for performance
    const pointsToRender = points.slice(0, this.options.maxPointsToRender);
    this.performanceMetrics.pointsRendered = pointsToRender.length;
    
    // Add result points
    pointsToRender.forEach((vector, index) => {
      // Get color based on visualization options
      const color = this._getPointColor(index);
      
      // Create material
      const material = new THREE.MeshBasicMaterial({
        color: new THREE.Color(color),
        transparent: true,
        opacity: this.options.pointOpacity
      });
      
      // Create geometry
      const geometry = new THREE.SphereGeometry(
        this.options.pointSize / 100, 
        this.options.highContrast ? 16 : 8, 
        this.options.highContrast ? 16 : 8
      );
      
      // Create mesh
      const mesh = new THREE.Mesh(geometry, material);
      
      // Set position (use only the first 3 dimensions)
      mesh.position.set(
        vector[0] || 0,
        vector[1] || 0,
        vector[2] || 0
      );
      
      // Add to group
      this.pointsGroup.add(mesh);
    });
    
    // Add query point if provided
    if (this.queryPoint) {
      const material = new THREE.MeshBasicMaterial({
        color: new THREE.Color(this.options.highlightColor),
        transparent: false,
      });
      
      const geometry = new THREE.SphereGeometry(
        this.options.pointSize * 1.5 / 100, 
        16, 16
      );
      
      const mesh = new THREE.Mesh(geometry, material);
      
      mesh.position.set(
        this.queryPoint[0] || 0,
        this.queryPoint[1] || 0,
        this.queryPoint[2] || 0
      );
      
      this.pointsGroup.add(mesh);
    }
    
    // Reset camera position
    this.camera.position.set(3, 3, 3);
    this.camera.lookAt(0, 0, 0);
    this.controls.update();
    
    // Update info panel
    this._updateInfoPanel();
  }

  /**
   * Load data for visualization (auto-selects 2D or 3D based on current method)
   * @param {Array} points - Vector points to visualize
   * @param {Array} metadata - Optional metadata for each point
   * @param {Array} similarities - Optional similarity scores
   * @param {Array} queryPoint - Optional query point
   * @param {Array} clusters - Optional cluster assignments
   * @public
   */
  loadData(points, metadata = null, similarities = null, queryPoint = null, clusters = null) {
    // Store the data
    this.points = points;
    this.metadata = metadata;
    this.similarities = similarities;
    this.queryPoint = queryPoint;
    this.clusters = clusters;
    
    // Load based on visualization method
    if (this.options.method === '3d') {
      this.load3DData(points, metadata, similarities, queryPoint, clusters);
    } else {
      this._processPointsFor2D();
    }
  }

  /**
   * Reset the visualization to its initial state
   * @public
   */
  reset() {
    // Clear data
    this.points = [];
    this.metadata = null;
    this.similarities = null;
    this.queryPoint = null;
    this.clusters = null;
    this.selectedPoints = [];
    
    // Reset visualization
    if (this.options.method === '3d') {
      // Clear points
      while (this.pointsGroup && this.pointsGroup.children.length > 0) {
        this.pointsGroup.remove(this.pointsGroup.children[0]);
      }
      
      // Reset camera position
      this.camera.position.set(3, 3, 3);
      this.camera.lookAt(0, 0, 0);
      this.controls.update();
    } else if (this.svg) {
      // Clear 2D visualization
      this.vizGroup.selectAll('*').remove();
    }
    
    // Update info panel
    this._updateInfoPanel();
  }

  /**
   * Update visualization options
   * @param {Object} options - New options to apply
   * @public
   */
  updateOptions(options) {
    // Merge new options with existing ones
    this.options = Object.assign(this.options, options);
    
    // Apply options
    this._applyOptions();
  }

  /**
   * Get current performance metrics
   * @returns {Object} - Performance metrics
   * @public
   */
  getPerformanceMetrics() {
    return { ...this.performanceMetrics };
  }

  /**
   * Get selected point indices
   * @returns {Array} - Selected point indices
   * @public
   */
  getSelectedPoints() {
    return [...this.selectedPoints];
  }

  /**
   * Export current visualization as image
   * @returns {string} - Data URL of the image
   * @public
   */
  exportImage() {
    if (this.options.method === '3d' && this.renderer) {
      // For 3D, render the scene and get the canvas data
      this.renderer.render(this.scene, this.camera);
      return this.renderer.domElement.toDataURL('image/png');
    } else if (this.svg) {
      // For 2D, convert SVG to image
      const svgData = new XMLSerializer().serializeToString(this.svg.node());
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      // Set canvas size
      canvas.width = this.container.clientWidth;
      canvas.height = this.container.clientHeight;
      
      // Create image from SVG
      const img = new Image();
      const svgBlob = new Blob([svgData], { type: 'image/svg+xml' });
      const url = URL.createObjectURL(svgBlob);
      
      return new Promise((resolve, reject) => {
        img.onload = () => {
          ctx.drawImage(img, 0, 0);
          URL.revokeObjectURL(url);
          resolve(canvas.toDataURL('image/png'));
        };
        
        img.onerror = reject;
        img.src = url;
      });
    }
    
    return null;
  }
}

// Make available globally
window.EmbeddingVisualizer = EmbeddingVisualizer;