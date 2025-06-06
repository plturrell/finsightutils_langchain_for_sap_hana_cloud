# Frontend for SAP HANA Cloud LangChain Integration

This directory contains the frontend UI for the SAP HANA Cloud LangChain Integration project. The frontend is built with modern web technologies and provides a responsive, accessible interface for interacting with the SAP HANA Cloud LangChain API.

## Features

- **Responsive Design**: Mobile-first approach for all screens and devices
- **Accessibility**: Dark mode, high contrast, screen reader support, and keyboard navigation
- **Interactive Visualizations**: 3D visualization of vector embeddings
- **User Authentication**: Secure login and authentication system
- **Error Handling**: Comprehensive error handling with helpful suggestions
- **API Integration**: Seamless integration with the backend API

## Running the Frontend

### Local Development

```bash
# Install dependencies
npm install

# Start the development server
npm start
```

The frontend will be available at http://localhost:3000.

### Building for Production

```bash
# Build the frontend
npm run build
```

The built files will be in the `build` directory.

## Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose -f docker-compose.frontend.yml up -d
```

### Vercel Deployment

The frontend is optimized for deployment on Vercel:

1. Push code to GitHub
2. Create a new project on Vercel
3. Configure build settings:
   - **Framework Preset**: Other
   - **Root Directory**: `frontend`
   - **Build Command**: `./vercel-build.sh`
   - **Output Directory**: `build`
4. Set environment variables:
   - `BACKEND_URL`: URL of your deployed backend API
5. Deploy

## Configuration

The frontend can be configured using environment variables:

- `BACKEND_URL`: URL of the backend API
- `VITE_APP_VERSION`: Application version
- `VITE_ENABLE_ANALYTICS`: Enable analytics features

These can be set in a `.env` file for local development or in the Vercel dashboard for production deployment.

## Browser Compatibility

The frontend is compatible with modern browsers:

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Technology Stack

- **React**: UI library
- **Bootstrap 5**: CSS framework
- **Three.js**: 3D visualization
- **JWT**: Authentication
- **Fetch API**: API communication