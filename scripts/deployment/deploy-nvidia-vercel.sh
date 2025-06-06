#!/bin/bash

# Enhanced Vercel Deployment Script for NVIDIA TensorRT Optimization
# This script deploys the frontend to Vercel with proper configuration for
# NVIDIA T4 GPU backend with TensorRT acceleration

# Exit on error
set -e

# ANSI color codes for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print script header
echo -e "${BLUE}===========================================================${NC}"
echo -e "${BLUE}    SAP HANA Cloud LangChain Integration - Vercel Deployment${NC}"
echo -e "${BLUE}    With NVIDIA TensorRT Optimization Support${NC}"
echo -e "${BLUE}===========================================================${NC}"

# Define variables with defaults that can be overridden by environment variables
BACKEND_URL=${BACKEND_URL:-"https://jupyter0-513syzm60.brevlab.com"}
PROJECT_ROOT=$(pwd)
VERCEL_PROJECT_NAME=${VERCEL_PROJECT_NAME:-"sap-hana-langchain-t4"}
ENVIRONMENT=${ENVIRONMENT:-"production"}
JWT_SECRET=${JWT_SECRET:-"sap-hana-langchain-t4-integration-secret-key-2025"}
TENSORRT_ENABLED=${TENSORRT_ENABLED:-"true"}
TENSORRT_CACHE_DIR=${TENSORRT_CACHE_DIR:-"/tmp/tensorrt_engines"}
TENSORRT_PRECISION=${TENSORRT_PRECISION:-"int8"}

# Print deployment configuration
echo -e "${YELLOW}Deployment Configuration:${NC}"
echo -e "  Backend URL: ${CYAN}$BACKEND_URL${NC}"
echo -e "  Environment: ${CYAN}$ENVIRONMENT${NC}"
echo -e "  Project Name: ${CYAN}$VERCEL_PROJECT_NAME${NC}"
echo -e "  TensorRT Enabled: ${CYAN}$TENSORRT_ENABLED${NC}"
echo -e "  TensorRT Precision: ${CYAN}$TENSORRT_PRECISION${NC}"
echo

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo -e "${YELLOW}Vercel CLI not found. Installing...${NC}"
    npm install -g vercel
fi

# Ensure we're in the project root
cd "$PROJECT_ROOT"

# Verify environment variables
if [ -z "$VERCEL_TOKEN" ]; then
    echo -e "${RED}Error: VERCEL_TOKEN is not set${NC}"
    echo -e "Please set the VERCEL_TOKEN environment variable by running:"
    echo -e "${CYAN}export VERCEL_TOKEN=your_vercel_token${NC}"
    exit 1
fi

# Update frontend configuration with the correct backend URL
echo -e "${YELLOW}Updating frontend configuration with backend URL: $BACKEND_URL${NC}"
sed -i.bak "s|const API_BASE_URL = '.*'|const API_BASE_URL = '$BACKEND_URL'|g" frontend/index.html

# Create config directories if they don't exist
mkdir -p config/vercel

# Create or update vercel.json with TensorRT configuration
echo -e "${YELLOW}Creating vercel.json configuration with TensorRT support...${NC}"
cat > config/vercel/vercel.json << EOFINNER
{
  "version": 2,
  "builds": [
    {
      "src": "api/vercel_integration.py",
      "use": "@vercel/python"
    },
    {
      "src": "frontend/**",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "api/vercel_integration.py"
    },
    {
      "src": "/(.*)",
      "dest": "frontend/\$1"
    }
  ],
  "env": {
    "T4_GPU_BACKEND_URL": "$BACKEND_URL",
    "ENVIRONMENT": "$ENVIRONMENT",
    "JWT_SECRET": "$JWT_SECRET",
    "TENSORRT_ENABLED": "$TENSORRT_ENABLED",
    "TENSORRT_PRECISION": "$TENSORRT_PRECISION",
    "TENSORRT_CACHE_DIR": "$TENSORRT_CACHE_DIR",
    "DEFAULT_TIMEOUT": "60"
  }
}
EOFINNER

# Create symlink to main vercel.json
rm -f vercel.json  # Remove existing file if it's not a symlink
ln -sf $(pwd)/config/vercel/vercel.json $(pwd)/vercel.json

# Create requirements-vercel.txt for Vercel Python if it doesn't exist
if [ ! -f "api/requirements-vercel.txt" ]; then
    echo -e "${YELLOW}Creating requirements-vercel.txt for Vercel deployment...${NC}"
    cat > api/requirements-vercel.txt << EOFINNER
fastapi==0.100.0
uvicorn==0.22.0
requests==2.31.0
pyjwt==2.8.0
pydantic==2.0.3
python-multipart==0.0.6
python-dotenv==1.0.0
EOFINNER
fi

# Create vercel.json symlink in the api directory for deployment
echo -e "${YELLOW}Creating vercel.json symlink in api directory...${NC}"
rm -f api/vercel.json  # Remove existing file if it's not a symlink
ln -sf $(pwd)/config/vercel/vercel.json $(pwd)/api/vercel.json

# Enhance the frontend index.html to better support TensorRT
echo -e "${YELLOW}Enhancing frontend for TensorRT visualization...${NC}"

# Create vector-visualization.js if it doesn't exist
if [ ! -f "frontend/vector-visualization.js" ]; then
    echo -e "${YELLOW}Creating vector visualization support for TensorRT...${NC}"
    mkdir -p frontend
    
    cat > frontend/vector-visualization.js << EOFINNER
/**
 * Vector Visualization for SAP HANA Cloud LangChain Integration
 * Optimized for TensorRT acceleration
 */

class VectorVisualization {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.options = Object.assign({
            height: '400px',
            colorScheme: 'default',
            highContrast: false,
            showLabels: true,
            colorblindMode: false,
            backgroundColor: '#f8f9fa',
            darkModeBackgroundColor: '#2a2a2a'
        }, options);
        
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.points = [];
        this.queryPoint = null;
        
        this.init();
    }
    
    init() {
        // Set container height
        this.container.style.height = this.options.height;
        
        // Initialize Three.js components
        this.scene = new THREE.Scene();
        
        // Set background color based on dark mode
        const isDarkMode = document.body.classList.contains('dark-mode');
        this.scene.background = new THREE.Color(
            isDarkMode ? this.options.darkModeBackgroundColor : this.options.backgroundColor
        );
        
        // Create camera
        this.camera = new THREE.PerspectiveCamera(
            75, this.container.clientWidth / this.container.clientHeight, 0.1, 1000
        );
        this.camera.position.z = 5;
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.container.appendChild(this.renderer.domElement);
        
        // Add orbit controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.25;
        
        // Add ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);
        
        // Add directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(0, 1, 0);
        this.scene.add(directionalLight);
        
        // Add window resize handler
        window.addEventListener('resize', this.onWindowResize.bind(this));
        
        // Start animation loop
        this.animate();
        
        // Add visualization info
        this.addInfo();
    }
    
    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
    
    animate() {
        requestAnimationFrame(this.animate.bind(this));
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    loadData(data) {
        // Clear previous visualization
        this.clearVisualization();
        
        // Add query point
        if (data.query_point) {
            this.addQueryPoint(data.query_point);
        }
        
        // Add result points
        if (data.points && data.points.length > 0) {
            this.addResultPoints(data.points, data.metadata, data.contents, data.similarities);
        }
        
        // Add axes for reference
        this.addAxes();
        
        // Reset camera position
        this.resetCamera();
    }
    
    clearVisualization() {
        // Remove all points from the scene
        while(this.scene.children.length > 0) { 
            this.scene.remove(this.scene.children[0]); 
        }
        this.points = [];
        this.queryPoint = null;
    }
    
    addQueryPoint(vector) {
        // Create material for query point
        const material = new THREE.MeshBasicMaterial({ 
            color: this.options.highContrast ? 0xff0000 : 0xff3366,
            size: 0.2
        });
        
        // Create geometry and mesh
        const geometry = new THREE.SphereGeometry(0.15, 32, 32);
        const mesh = new THREE.Mesh(geometry, material);
        
        // Set position
        mesh.position.set(vector[0], vector[1], vector[2]);
        
        // Add to scene
        this.scene.add(mesh);
        this.queryPoint = mesh;
        
        // Add label
        if (this.options.showLabels) {
            this.addLabel("Query", vector, true);
        }
    }
    
    addResultPoints(vectors, metadata, contents, similarities) {
        vectors.forEach((vector, index) => {
            // Determine color based on similarity
            const similarity = similarities[index] || 0.5;
            const color = this.getColorBySimilarity(similarity);
            
            // Create material
            const material = new THREE.MeshBasicMaterial({ 
                color: color,
                size: 0.1
            });
            
            // Create geometry and mesh
            const geometry = new THREE.SphereGeometry(0.1, 16, 16);
            const mesh = new THREE.Mesh(geometry, material);
            
            // Set position
            mesh.position.set(vector[0], vector[1], vector[2]);
            
            // Add to scene
            this.scene.add(mesh);
            this.points.push(mesh);
            
            // Add label
            if (this.options.showLabels) {
                const label = metadata[index]?.title || \`Result \${index + 1}\`;
                this.addLabel(label, vector, false);
            }
        });
    }
    
    addLabel(text, position, isQuery = false) {
        // Create div for label
        const labelDiv = document.createElement('div');
        labelDiv.className = 'vector-label';
        labelDiv.textContent = text;
        labelDiv.style.position = 'absolute';
        labelDiv.style.fontSize = isQuery ? '14px' : '12px';
        labelDiv.style.fontWeight = isQuery ? 'bold' : 'normal';
        labelDiv.style.color = isQuery ? '#ff3366' : '#666';
        labelDiv.style.padding = '2px 5px';
        labelDiv.style.backgroundColor = 'rgba(255, 255, 255, 0.7)';
        labelDiv.style.borderRadius = '3px';
        labelDiv.style.pointerEvents = 'none';
        
        // Add to container
        this.container.appendChild(labelDiv);
        
        // Update position in animation loop
        const updatePosition = () => {
            // Project 3D position to 2D screen coordinates
            const vector = new THREE.Vector3(position[0], position[1], position[2]);
            vector.project(this.camera);
            
            // Convert to CSS coordinates
            const x = (vector.x * 0.5 + 0.5) * this.container.clientWidth;
            const y = (-vector.y * 0.5 + 0.5) * this.container.clientHeight;
            
            // Update label position
            labelDiv.style.left = \`\${x}px\`;
            labelDiv.style.top = \`\${y}px\`;
            
            // Check if label is in front of camera
            labelDiv.style.display = vector.z < 1 ? 'block' : 'none';
            
            // Continue updating
            requestAnimationFrame(updatePosition);
        };
        
        updatePosition();
    }
    
    addAxes() {
        // Create axes
        const axesHelper = new THREE.AxesHelper(2);
        this.scene.add(axesHelper);
        
        // Add axis labels
        this.addLabel("X", [2.2, 0, 0], false);
        this.addLabel("Y", [0, 2.2, 0], false);
        this.addLabel("Z", [0, 0, 2.2], false);
    }
    
    resetCamera() {
        this.camera.position.set(3, 3, 3);
        this.camera.lookAt(0, 0, 0);
        this.controls.update();
    }
    
    getColorBySimilarity(similarity) {
        // Color scheme based on similarity score
        if (this.options.colorblindMode) {
            // Colorblind-friendly colors
            if (similarity > 0.8) return 0x0072B2; // Blue
            if (similarity > 0.6) return 0x56B4E9; // Light blue
            if (similarity > 0.4) return 0xCC79A7; // Pink
            if (similarity > 0.2) return 0xF0E442; // Yellow
            return 0xD55E00; // Orange
        } else {
            // Standard color scheme
            if (similarity > 0.8) return 0x28a745; // Green
            if (similarity > 0.6) return 0x17a2b8; // Teal
            if (similarity > 0.4) return 0x007bff; // Blue
            if (similarity > 0.2) return 0xffc107; // Yellow
            return 0xdc3545; // Red
        }
    }
    
    addInfo() {
        // Create info panel
        const infoPanel = document.createElement('div');
        infoPanel.className = 'vector-vis-info';
        infoPanel.style.position = 'absolute';
        infoPanel.style.bottom = '10px';
        infoPanel.style.left = '10px';
        infoPanel.style.backgroundColor = 'rgba(255, 255, 255, 0.7)';
        infoPanel.style.padding = '5px 10px';
        infoPanel.style.borderRadius = '5px';
        infoPanel.style.fontSize = '12px';
        infoPanel.style.color = '#333';
        
        infoPanel.innerHTML = \`
            <div><strong>Controls:</strong></div>
            <div>Left click + drag: Rotate</div>
            <div>Right click + drag: Pan</div>
            <div>Scroll: Zoom</div>
        \`;
        
        this.container.appendChild(infoPanel);
    }
}
EOFINNER
fi

# Create CSS file for vector visualization
if [ ! -f "frontend/vector-visualization.css" ]; then
    echo -e "${YELLOW}Creating CSS for vector visualization...${NC}"
    cat > frontend/vector-visualization.css << EOFINNER
.vector-label {
    user-select: none;
    z-index: 10;
    transition: opacity 0.2s ease;
}

.vector-vis-info {
    user-select: none;
    z-index: 20;
    transition: opacity 0.2s ease;
}

#vector-visualization {
    position: relative;
    width: 100%;
    height: 400px;
    border-radius: 5px;
    overflow: hidden;
}

@media (prefers-color-scheme: dark) {
    .vector-label {
        color: #e0e0e0 !important;
        background-color: rgba(50, 50, 50, 0.7) !important;
    }
    
    .vector-vis-info {
        color: #e0e0e0 !important;
        background-color: rgba(50, 50, 50, 0.7) !important;
    }
}
EOFINNER
fi

# Create build script for Vercel
echo -e "${YELLOW}Creating build script for Vercel...${NC}"
cat > frontend/vercel-build.sh << EOFINNER
#!/bin/bash
echo "Building SAP HANA Cloud LangChain Integration with TensorRT optimization"
cp -r frontend/* .
EOFINNER

chmod +x frontend/vercel-build.sh

# Create a .vercelignore file
echo -e "${YELLOW}Creating .vercelignore file...${NC}"
cat > .vercelignore << EOFINNER
README.md
README_DOCKER.md
NVIDIA_DEPLOYMENT.md
NVIDIA_T4_VERCEL_DEPLOYMENT.md
DEPLOYMENT_SUMMARY.md
.git
__pycache__
*.pyc
*.pyo
.DS_Store
.env
.env.*
langchain_hana
tests
notebooks
scripts
*.sh
!frontend/vercel-build.sh
docker-compose.*
Dockerfile*
node_modules
.archive/
config/
docs/
examples/
EOFINNER

# Create a simple vercel.json in the api directory for deployment 
echo -e "${YELLOW}Creating vercel.json in api directory...${NC}"
cat > api/vercel.json << EOFINNER
{
  "version": 2,
  "public": false,
  "installCommand": "pip install -r requirements-vercel.txt",
  "builds": [
    {
      "src": "vercel_integration.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "vercel_integration.py"
    }
  ],
  "env": {
    "T4_GPU_BACKEND_URL": "$BACKEND_URL",
    "ENVIRONMENT": "$ENVIRONMENT",
    "JWT_SECRET": "$JWT_SECRET",
    "TENSORRT_ENABLED": "$TENSORRT_ENABLED",
    "TENSORRT_PRECISION": "$TENSORRT_PRECISION",
    "TENSORRT_CACHE_DIR": "$TENSORRT_CACHE_DIR",
    "DEFAULT_TIMEOUT": "60"
  }
}
EOFINNER

# Deploy to Vercel
echo -e "${GREEN}Deploying to Vercel...${NC}"
if [ "$ENVIRONMENT" = "production" ]; then
    # Production deployment
    vercel --prod --confirm --token "$VERCEL_TOKEN" --name "$VERCEL_PROJECT_NAME"
else
    # Preview deployment
    vercel --confirm --token "$VERCEL_TOKEN" --name "$VERCEL_PROJECT_NAME"
fi

# Get the deployment URL
DEPLOYMENT_URL=$(vercel --token "$VERCEL_TOKEN" ls "$VERCEL_PROJECT_NAME" -j | jq -r '.deployments[0].url')
echo -e "${GREEN}Deployment successful! Your application is available at: https://$DEPLOYMENT_URL${NC}"

# Save the deployment URL to a file for reference
echo "https://$DEPLOYMENT_URL" > deployment_url.txt
echo -e "${BLUE}Deployment URL saved to deployment_url.txt${NC}"

# Print connection information
echo ""
echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}Frontend-Backend Connection Information${NC}"
echo -e "${BLUE}=======================================${NC}"
echo -e "${YELLOW}Frontend URL:${NC} https://$DEPLOYMENT_URL"
echo -e "${YELLOW}Backend URL:${NC} $BACKEND_URL"
echo -e "${YELLOW}TensorRT Enabled:${NC} $TENSORRT_ENABLED"
echo -e "${YELLOW}TensorRT Precision:${NC} $TENSORRT_PRECISION"
echo ""
echo -e "${YELLOW}To test the connection, visit:${NC}"
echo -e "https://$DEPLOYMENT_URL"
echo ""
echo -e "${YELLOW}If you need to update the backend URL, run:${NC}"
echo -e "${CYAN}BACKEND_URL=<new-url> ./scripts/deployment/deploy-nvidia-vercel.sh${NC}"
echo -e "${BLUE}=======================================${NC}"

# Show TensorRT instructions
echo ""
echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}NVIDIA TensorRT Optimization Information${NC}"
echo -e "${BLUE}=======================================${NC}"
echo -e "${YELLOW}TensorRT status:${NC} ${GREEN}Enabled${NC}"
echo -e "${YELLOW}When using the embeddings API:${NC}"
echo -e "1. Make sure the 'Use TensorRT' checkbox is checked"
echo -e "2. Select the appropriate precision level for your needs:"
echo -e "   - ${CYAN}INT8${NC}: Fastest inference, slightly lower accuracy"
echo -e "   - ${CYAN}FP16${NC}: Balance of speed and accuracy"
echo -e "   - ${CYAN}FP32${NC}: Highest accuracy, slowest inference"
echo -e ""
echo -e "${YELLOW}Backend optimizations:${NC}"
echo -e "- TensorRT engine caching is enabled at: ${CYAN}$TENSORRT_CACHE_DIR${NC}"
echo -e "- Default precision is set to: ${CYAN}$TENSORRT_PRECISION${NC}"
echo -e "${BLUE}=======================================${NC}"