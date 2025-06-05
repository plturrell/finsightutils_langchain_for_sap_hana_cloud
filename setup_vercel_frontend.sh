#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}Setup Vercel Frontend for SAP HANA Cloud LangChain Integration${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is not installed. Please install Node.js first.${NC}"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo -e "${RED}Error: npm is not installed. Please install npm first.${NC}"
    exit 1
fi

# Check if Vercel CLI is installed
if command -v vercel &> /dev/null; then
    echo -e "${GREEN}Vercel CLI is installed.${NC}"
    VERCEL_CLI_INSTALLED=true
else
    echo -e "${YELLOW}WARNING: Vercel CLI is not installed.${NC}"
    echo -e "${YELLOW}You can still prepare the frontend for manual deployment.${NC}"
    VERCEL_CLI_INSTALLED=false
    
    read -p "Would you like to install Vercel CLI? (y/n) " install_vercel
    if [[ "$install_vercel" =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Installing Vercel CLI...${NC}"
        npm install -g vercel
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Vercel CLI installed successfully!${NC}"
            VERCEL_CLI_INSTALLED=true
        else
            echo -e "${RED}Failed to install Vercel CLI.${NC}"
        fi
    fi
fi

# Set up environment variables for frontend
echo -e "${BLUE}Setting up environment variables for frontend...${NC}"
read -p "Enter the URL of your NVIDIA backend (e.g., http://your-server:8000): " BACKEND_URL
BACKEND_URL=${BACKEND_URL:-http://localhost:8000}

cat > frontend/.env.production << EOL
# Backend API URL
VITE_API_BASE_URL=${BACKEND_URL}

# App Version
VITE_APP_VERSION=1.0.0

# Feature flags
VITE_ENABLE_VECTOR_VISUALIZATION=true
VITE_ENABLE_DARK_MODE=true
VITE_ENABLE_ACCESSIBILITY=true
EOL

echo -e "${GREEN}Environment file created: frontend/.env.production${NC}"

# Prepare frontend build
echo -e "${BLUE}Preparing frontend build...${NC}"
cd frontend

# Install dependencies
echo -e "${BLUE}Installing frontend dependencies...${NC}"
npm install --force || npm install --legacy-peer-deps

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Frontend dependencies installed successfully!${NC}"
else
    echo -e "${RED}Failed to install frontend dependencies.${NC}"
    exit 1
fi

# Test build
echo -e "${BLUE}Testing frontend build...${NC}"
npm run build

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Frontend build successful!${NC}"
else
    echo -e "${RED}Frontend build failed. Please check the error messages above.${NC}"
    exit 1
fi

# Vercel deployment
if [ "$VERCEL_CLI_INSTALLED" = true ]; then
    echo -e "${BLUE}Would you like to deploy to Vercel now? (y/n)${NC}"
    read -p "Deploy to Vercel? " deploy_decision
    
    if [[ "$deploy_decision" =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Deploying to Vercel...${NC}"
        
        # Check if logged in to Vercel
        vercel whoami &> /dev/null
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}You need to log in to Vercel.${NC}"
            vercel login
        fi
        
        # Deploy to Vercel
        vercel --prod
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Successfully deployed to Vercel!${NC}"
        else
            echo -e "${RED}Failed to deploy to Vercel.${NC}"
        fi
    else
        echo -e "${YELLOW}Skipping Vercel deployment.${NC}"
    fi
else
    echo -e "${YELLOW}Vercel CLI not available. Manual deployment instructions:${NC}"
    echo -e "1. Create a new project on Vercel (https://vercel.com/new)"
    echo -e "2. Connect to your GitHub repository"
    echo -e "3. Configure the following settings:"
    echo -e "   - Framework Preset: Other"
    echo -e "   - Root Directory: frontend"
    echo -e "   - Build Command: ./vercel-build.sh"
    echo -e "   - Output Directory: build"
    echo -e "4. Add the environment variable:"
    echo -e "   - BACKEND_URL=${BACKEND_URL}"
    echo -e "5. Deploy the project"
fi

# Return to the project root
cd ..

# Create a configuration file for connecting to the deployed frontend
echo -e "${BLUE}Creating frontend configuration file...${NC}"
cat > frontend_config.json << EOL
{
  "backend_url": "${BACKEND_URL}",
  "frontend_url": "https://your-deployed-frontend-url.vercel.app"
}
EOL

echo -e "${GREEN}Frontend configuration file created: frontend_config.json${NC}"
echo -e "${YELLOW}Please update the 'frontend_url' in frontend_config.json after deployment.${NC}"

echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${BLUE}=========================================================${NC}"
echo -e "To test the frontend locally:"
echo -e "  1. Navigate to the frontend directory: ${YELLOW}cd frontend${NC}"
echo -e "  2. Start the development server: ${YELLOW}npm start${NC}"
echo -e ""
echo -e "The frontend will be available at: ${CYAN}http://localhost:3000${NC}"
echo -e ""
echo -e "To deploy to Vercel manually:"
echo -e "  Follow the instructions in the Vercel dashboard or use:"
echo -e "  ${YELLOW}cd frontend && vercel --prod${NC}"
echo -e ""
echo -e "After deployment, update the frontend URL in ${YELLOW}frontend_config.json${NC}"
echo -e "${BLUE}=========================================================${NC}"