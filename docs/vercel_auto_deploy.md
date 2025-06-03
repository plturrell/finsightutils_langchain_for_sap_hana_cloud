# Setting Up Automatic Deployments from GitHub to Vercel

This guide provides step-by-step instructions for connecting your GitHub repository to Vercel for automatic deployments.

## Prerequisites

- GitHub repository with your project code
- Vercel account
- Owner or admin access to both accounts

## Steps to Connect GitHub to Vercel

### 1. Create a Vercel Account

If you don't already have one:

1. Go to [Vercel's website](https://vercel.com)
2. Click "Sign Up"
3. Choose "Continue with GitHub" to link your GitHub account

### 2. Import Your Repository

1. From the Vercel dashboard, click "Add New..." → "Project"
2. Select the "Import Git Repository" option
3. Choose your GitHub account
4. Find and select `langchain-integration-for-sap-hana-cloud` repository
5. Click "Import"

### 3. Configure Project Settings

1. **Framework Preset**: Select "Other"
2. **Root Directory**: Leave as default (project root)
3. **Build Command**: `pip install -r api/requirements-vercel.txt`
4. **Output Directory**: Leave blank
5. **Install Command**: `pip install -r api/requirements-vercel.txt`

### 4. Environment Variables

Add your environment variables:

1. Scroll down to "Environment Variables" section
2. Add each variable from your `.env.vercel` file:
   - `HANA_HOST`
   - `HANA_PORT`
   - `HANA_USER`
   - `HANA_PASSWORD`
   - `LOG_LEVEL`
   - `EMBEDDING_MODEL`

3. Mark sensitive variables as "Encrypted"

### 5. Deploy Settings

1. Click "Deploy"
2. Vercel will build and deploy your project
3. When complete, you'll receive a deployment URL

## Configuring Automatic Deployments

By default, Vercel will automatically deploy:

1. **Production Deployments**: When you push to the `main` branch
2. **Preview Deployments**: When you create pull requests

To customize these settings:

1. Go to your project on Vercel dashboard
2. Navigate to "Settings" → "Git"
3. Under "Production Branch", confirm `main` is selected
4. Under "Ignored Build Step", you can add conditions to skip builds if needed

## Testing the Automatic Deployment

To test the automatic deployment:

1. Make a small change to your project
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Test automatic deployment to Vercel"
   git push
   ```
3. Go to your Vercel dashboard to see the deployment progress
4. Once complete, visit your deployment URL to verify changes

## Troubleshooting

If your automatic deployments aren't working:

1. **Check Repository Connection**:
   - In Vercel, go to Settings → Git → Connected Git Repository
   - Verify the correct repository is connected

2. **Check Build Logs**:
   - Review deployment logs for errors
   - Common issues include missing dependencies or environment variables

3. **Check Permissions**:
   - Ensure Vercel has appropriate access to your GitHub repository
   - You may need to reinstall the Vercel GitHub App with correct permissions

4. **Check GitHub Webhooks**:
   - Go to your GitHub repository → Settings → Webhooks
   - Verify a Vercel webhook exists and is active

## Managing Deployments

From your Vercel dashboard, you can:

1. **View Deployment History**: See all past deployments
2. **Roll Back**: Revert to previous deployments if needed
3. **Promote Preview**: Promote preview deployments to production
4. **Inspect**: View detailed logs and analytics

## Next Steps

After setting up automatic deployments:

1. Configure a custom domain
2. Set up monitoring and alerts
3. Implement API authentication
4. Consider setting up staging environments