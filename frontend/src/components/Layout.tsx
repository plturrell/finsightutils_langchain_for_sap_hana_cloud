import React, { useState } from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import {
  AppBar,
  Box,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  useMediaQuery,
  useTheme,
  Divider,
  Button,
  Avatar,
  Menu,
  MenuItem,
  Tooltip,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Search as SearchIcon,
  Speed as SpeedIcon,
  Settings as SettingsIcon,
  ChevronLeft as ChevronLeftIcon,
  Person as PersonIcon,
  Logout as LogoutIcon,
  DarkMode as DarkModeIcon,
  LightMode as LightModeIcon,
  Notifications as NotificationsIcon,
  Help as HelpIcon,
  AccountTree as AccountTreeIcon,
} from '@mui/icons-material';
import SAPLogo from './SAPLogo';
import ErrorHandler from './ErrorHandler';
import { useError } from '../context/ErrorContext';
import HumanText from './HumanText';

const drawerWidth = 280;

const navItems = [
  { name: 'Discoveries', path: '/', icon: <DashboardIcon /> },
  { name: 'Find Answers', path: '/search', icon: <SearchIcon /> },
  { name: 'Magic Behind', path: '/benchmark', icon: <SpeedIcon /> },
  { name: 'Create', path: '/developer', icon: <AccountTreeIcon /> },
  { name: 'Personalize', path: '/settings', icon: <SettingsIcon /> },
];

const Layout: React.FC = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [mobileOpen, setMobileOpen] = useState(false);
  const [userMenuAnchor, setUserMenuAnchor] = useState<null | HTMLElement>(null);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleUserMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setUserMenuAnchor(event.currentTarget);
  };

  const handleUserMenuClose = () => {
    setUserMenuAnchor(null);
  };

  const drawer = (
    <>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          padding: 2,
          paddingLeft: 3,
        }}
      >
        <SAPLogo height={32} />
        <HumanText
          variant="h6"
          sx={{
            ml: 1.5,
            fontWeight: 600,
            background: 'linear-gradient(90deg, #0066B3 0%, #19B5FE 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          Knowledge Explorer
        </HumanText>
        {isMobile && (
          <IconButton
            onClick={handleDrawerToggle}
            sx={{ ml: 'auto', color: 'text.secondary' }}
          >
            <ChevronLeftIcon />
          </IconButton>
        )}
      </Box>

      <Divider sx={{ opacity: 0.6 }} />

      <List sx={{ pt: 2 }}>
        {navItems.map((item) => (
          <ListItem key={item.name} disablePadding>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => {
                navigate(item.path);
                if (isMobile) setMobileOpen(false);
              }}
              sx={{
                borderRadius: '0 20px 20px 0',
                marginRight: 2,
                marginLeft: 0,
                paddingLeft: 3,
                position: 'relative',
                '&.Mui-selected': {
                  backgroundColor: 'rgba(0, 102, 179, 0.1)',
                  '&:before': {
                    content: '""',
                    position: 'absolute',
                    left: 0,
                    top: '25%',
                    height: '50%',
                    width: 4,
                    backgroundColor: theme.palette.primary.main,
                    borderRadius: '0 4px 4px 0',
                  },
                },
                '&:hover': {
                  backgroundColor: 'rgba(0, 102, 179, 0.05)',
                },
              }}
            >
              <ListItemIcon
                sx={{
                  color: location.pathname === item.path
                    ? theme.palette.primary.main
                    : theme.palette.text.secondary,
                  minWidth: 40,
                }}
              >
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={<HumanText>{item.name}</HumanText>}
                primaryTypographyProps={{
                  fontWeight: location.pathname === item.path ? 500 : 400,
                  color: location.pathname === item.path
                    ? theme.palette.primary.main
                    : theme.palette.text.primary,
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Box sx={{ flexGrow: 1 }} />

      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Button
          variant="outlined"
          startIcon={<HelpIcon />}
          fullWidth
          sx={{ borderRadius: 3, fontWeight: 500 }}
        >
          <HumanText>Documentation</HumanText>
        </Button>
      </Box>
    </>
  );

  const { error, clearError } = useError();

  return (
    <Box sx={{ display: 'flex', height: '100%' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          width: { md: `calc(100% - ${drawerWidth}px)` },
          ml: { md: `${drawerWidth}px` },
          borderRadius: 0,
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { md: 'none' }, color: 'text.primary' }}
          >
            <MenuIcon />
          </IconButton>

          <HumanText 
            variant="h6" 
            component="div" 
            sx={{ 
              flexGrow: 1, 
              fontWeight: 500,
              color: theme.palette.text.primary,
              display: { xs: 'none', sm: 'block' }
            }}
          >
            {navItems.find(item => item.path === location.pathname)?.name || 'Dashboard'}
          </HumanText>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Tooltip title="Toggle theme">
              <IconButton sx={{ color: 'text.secondary' }}>
                <LightModeIcon />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Notifications">
              <IconButton sx={{ color: 'text.secondary' }}>
                <NotificationsIcon />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Account">
              <IconButton
                onClick={handleUserMenuOpen}
                sx={{ p: 0.5 }}
              >
                <Avatar
                  sx={{
                    width: 40,
                    height: 40,
                    border: `2px solid ${theme.palette.primary.main}`,
                    bgcolor: theme.palette.primary.light,
                  }}
                >
                  <PersonIcon />
                </Avatar>
              </IconButton>
            </Tooltip>
            
            <Menu
              anchorEl={userMenuAnchor}
              open={Boolean(userMenuAnchor)}
              onClose={handleUserMenuClose}
              PaperProps={{
                elevation: 3,
                sx: {
                  mt: 1,
                  borderRadius: 2,
                  minWidth: 180,
                }
              }}
            >
              <MenuItem onClick={handleUserMenuClose}>
                <ListItemIcon>
                  <PersonIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText primary={<HumanText>Profile</HumanText>} />
              </MenuItem>
              <MenuItem onClick={handleUserMenuClose}>
                <ListItemIcon>
                  <SettingsIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText primary={<HumanText>Account Settings</HumanText>} />
              </MenuItem>
              <Divider />
              <MenuItem onClick={handleUserMenuClose}>
                <ListItemIcon>
                  <LogoutIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText primary={<HumanText>Logout</HumanText>} />
              </MenuItem>
            </Menu>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Drawer */}
      <Box
        component="nav"
        sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
      >
        {/* Mobile drawer */}
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better mobile performance
          }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
              borderRadius: '0 16px 16px 0',
              boxShadow: theme.shadows[8],
            },
          }}
        >
          {drawer}
        </Drawer>

        {/* Desktop drawer */}
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', md: 'block' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
              borderColor: 'divider',
              backgroundImage: 'linear-gradient(rgba(255, 255, 255, 0.9), rgba(245, 247, 250, 0.9))',
            },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>

      {/* Main content with enhanced white space and material consistency */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { md: `calc(100% - ${drawerWidth}px)` },
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          bgcolor: 'background.default',
          position: 'relative',
        }}
      >
        <Toolbar /> {/* Spacer for AppBar */}
        <Box 
          sx={{ 
            p: { xs: 2, sm: 3, md: 4, lg: 5 },  // Enhanced responsive padding
            flexGrow: 1, 
            overflowY: 'auto',
            // Refined subtle background pattern with reduced opacity for better white space
            backgroundImage: 'radial-gradient(circle at 25px 25px, rgba(0, 102, 179, 0.02) 2%, transparent 0%), radial-gradient(circle at 75px 75px, rgba(0, 102, 179, 0.02) 2%, transparent 0%)',
            backgroundSize: '100px 100px',
            // Smoother, more subtle shadow at the top edge
            boxShadow: 'inset 0 6px 6px -6px rgba(0, 0, 0, 0.04)',
            // Add subtle gradient to enhance depth perception
            '&:after': {
              content: '""',
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              height: '120px',
              background: 'linear-gradient(180deg, rgba(245, 247, 250, 0.6) 0%, rgba(245, 247, 250, 0) 100%)',
              pointerEvents: 'none',
              opacity: 0.7,
              zIndex: 0,
            },
            // Improved scrollbar styling for better material consistency
            '&::-webkit-scrollbar': {
              width: '6px',
              backgroundColor: 'transparent',
            },
            '&::-webkit-scrollbar-thumb': {
              backgroundColor: 'rgba(0, 0, 0, 0.1)',
              borderRadius: '6px',
              '&:hover': {
                backgroundColor: 'rgba(0, 0, 0, 0.2)',
              },
            },
            scrollbarWidth: 'thin',
            scrollbarColor: 'rgba(0, 0, 0, 0.1) transparent',
          }}
        >
          {error && (
            <Box sx={{ 
              mb: { xs: 3, sm: 4, md: 5 }, 
              maxWidth: 1200, 
              mx: 'auto',
              position: 'relative',
              zIndex: 1,
            }}>
              <ErrorHandler error={error} onClose={clearError} />
            </Box>
          )}
          <Box 
            sx={{ 
              maxWidth: 1200, 
              mx: 'auto',
              position: 'relative',
              zIndex: 1,
              // Enhanced animated fade-in effect for content
              animation: 'fadeIn 0.4s ease-out',
              '@keyframes fadeIn': {
                from: { opacity: 0, transform: 'translateY(10px)' },
                to: { opacity: 1, transform: 'translateY(0)' }
              },
              // Add additional breathing room at the bottom
              pb: { xs: 2, sm: 3, md: 4 },
            }}
          >
            <Outlet />
          </Box>
        </Box>
      </Box>
    </Box>
  );
};

export default Layout;