import React, { useState, useEffect } from 'react';
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
  alpha,
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
  Psychology as PsychologyIcon,
  Storage as StorageIcon,
  BlurOn as BlurOnIcon,
  TravelExplore as TravelExploreIcon,
} from '@mui/icons-material';
import { useSpring, animated, config, useTrail, useChain, useSpringRef } from '@react-spring/web';
import SAPLogo from './SAPLogo';
import ErrorHandler from './ErrorHandler';
import { useError } from '../context/ErrorContext';
import HumanText from './HumanText';

// Create animated versions of MUI components
const AnimatedAppBar = animated(AppBar);
const AnimatedToolbar = animated(Toolbar);
const AnimatedBox = animated(Box);
const AnimatedIconButton = animated(IconButton);
const AnimatedListItem = animated(ListItem);
const AnimatedListItemText = animated(ListItemText);
const AnimatedAvatar = animated(Avatar);
const AnimatedButton = animated(Button);
const AnimatedTypography = animated(Typography);

const drawerWidth = 280;

const navItems = [
  { name: 'Discoveries', path: '/', icon: <DashboardIcon /> },
  { name: 'Find Answers', path: '/search', icon: <SearchIcon /> },
  { name: 'Reasoning', path: '/reasoning', icon: <PsychologyIcon /> },
  { name: 'Data Pipeline', path: '/data-pipeline', icon: <StorageIcon /> },
  { name: 'Vector Creation', path: '/vector-creation', icon: <BlurOnIcon /> },
  { name: 'Vector Exploration', path: '/vector-exploration', icon: <TravelExploreIcon /> },
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
  const [layoutVisible, setLayoutVisible] = useState(false);

  // Animation spring refs for chaining
  const appBarSpringRef = useSpringRef();
  const logoSpringRef = useSpringRef();
  const navItemsSpringRef = useSpringRef();
  const contentSpringRef = useSpringRef();

  // AppBar animation
  const appBarAnimation = useSpring({
    ref: appBarSpringRef,
    from: { opacity: 0, transform: 'translateY(-20px)' },
    to: { opacity: layoutVisible ? 1 : 0, transform: layoutVisible ? 'translateY(0)' : 'translateY(-20px)' },
    config: { tension: 280, friction: 60 }
  });

  // Logo animation
  const logoAnimation = useSpring({
    ref: logoSpringRef,
    from: { opacity: 0, transform: 'scale(0.8)' },
    to: { opacity: layoutVisible ? 1 : 0, transform: layoutVisible ? 'scale(1)' : 'scale(0.8)' },
    config: { tension: 280, friction: 60 }
  });

  // Theme toggle animation
  const themeToggleAnimation = useSpring({
    transform: 'rotate(0deg)',
    config: { tension: 300, friction: 10 }
  });

  // Avatar animation
  const avatarAnimation = useSpring({
    from: { transform: 'scale(0.8)', opacity: 0 },
    to: { transform: layoutVisible ? 'scale(1)' : 'scale(0.8)', opacity: layoutVisible ? 1 : 0 },
    config: { tension: 200, friction: 20 }
  });
  
  // Notification bell animation
  const bellAnimation = useSpring({
    from: { transform: 'rotate(-30deg)', opacity: 0 },
    to: { transform: layoutVisible ? 'rotate(0deg)' : 'rotate(-30deg)', opacity: layoutVisible ? 1 : 0 },
    config: { tension: 200, mass: 1, friction: 20 }
  });

  // Content animation
  const contentAnimation = useSpring({
    ref: contentSpringRef,
    from: { opacity: 0, transform: 'translateY(30px)' },
    to: { opacity: layoutVisible ? 1 : 0, transform: layoutVisible ? 'translateY(0px)' : 'translateY(30px)' },
    config: { tension: 280, friction: 60 }
  });

  // Navigation items trail animation
  const navItemsTrail = useTrail(navItems.length, {
    ref: navItemsSpringRef,
    from: { opacity: 0, x: -20 },
    to: { opacity: layoutVisible ? 1 : 0, x: layoutVisible ? 0 : -20 },
    config: { mass: 1, tension: 280, friction: 60 }
  });

  // Animation sequence
  useChain(
    layoutVisible 
      ? [appBarSpringRef, logoSpringRef, navItemsSpringRef, contentSpringRef] 
      : [contentSpringRef, navItemsSpringRef, logoSpringRef, appBarSpringRef],
    layoutVisible 
      ? [0, 0.2, 0.3, 0.4] 
      : [0, 0.1, 0.2, 0.3]
  );

  // Set layoutVisible to true after component mounts
  useEffect(() => {
    const timer = setTimeout(() => {
      setLayoutVisible(true);
    }, 150);
    return () => clearTimeout(timer);
  }, []);

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
      <AnimatedBox
        style={logoAnimation}
        sx={{
          display: 'flex',
          alignItems: 'center',
          padding: 2,
          paddingLeft: 3,
        }}
      >
        <SAPLogo height={32} />
        <AnimatedTypography
          variant="h6"
          style={{
            background: 'linear-gradient(90deg, #0066B3 0%, #19B5FE 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundSize: '200% 100%',
            backgroundPosition: 'right bottom',
            ...useSpring({
              from: { backgroundPosition: '0% 50%' },
              to: { backgroundPosition: '100% 50%' },
              config: { duration: 3000 },
              loop: { reverse: true }
            })
          }}
          sx={{
            ml: 1.5,
            fontWeight: 600,
          }}
        >
          Knowledge Explorer
        </AnimatedTypography>
        {isMobile && (
          <AnimatedIconButton
            onClick={handleDrawerToggle}
            style={useSpring({
              transform: mobileOpen ? 'rotate(0deg)' : 'rotate(180deg)',
              config: { tension: 200, friction: 20 }
            })}
            sx={{ 
              ml: 'auto', 
              color: 'text.secondary',
              '&:hover': {
                color: theme.palette.primary.main,
                backgroundColor: alpha(theme.palette.primary.main, 0.08),
              }
            }}
          >
            <ChevronLeftIcon />
          </AnimatedIconButton>
        )}
      </AnimatedBox>

      <Divider sx={{ opacity: 0.6 }} />

      <List sx={{ pt: 2 }}>
        {navItemsTrail.map((style, index) => {
          const item = navItems[index];
          const isSelected = location.pathname === item.path;
          
          return (
            <AnimatedListItem key={item.name} style={style} disablePadding>
              <ListItemButton
                selected={isSelected}
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
                  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                  '&.Mui-selected': {
                    backgroundColor: 'rgba(0, 102, 179, 0.1)',
                    transform: 'translateX(4px)',
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
                    transform: 'translateX(2px)',
                  },
                }}
              >
                <ListItemIcon
                  sx={{
                    color: isSelected
                      ? theme.palette.primary.main
                      : theme.palette.text.secondary,
                    minWidth: 40,
                    transition: 'transform 0.2s ease-in-out',
                    transform: isSelected ? 'scale(1.2)' : 'scale(1)',
                    '& svg': {
                      transition: 'transform 0.3s ease',
                    },
                    '&:hover svg': {
                      transform: 'rotate(5deg)',
                    }
                  }}
                >
                  {item.icon}
                </ListItemIcon>
                <AnimatedListItemText
                  primary={<HumanText>{item.name}</HumanText>}
                  primaryTypographyProps={{
                    fontWeight: isSelected ? 600 : 400,
                    color: isSelected
                      ? theme.palette.primary.main
                      : theme.palette.text.primary,
                    sx: {
                      transition: 'all 0.3s ease',
                      transform: isSelected ? 'translateX(2px)' : 'translateX(0)',
                      ...(isSelected && {
                        background: 'linear-gradient(90deg, #0066B3 0%, #19B5FE 100%)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                      })
                    }
                  }}
                />
              </ListItemButton>
            </AnimatedListItem>
          );
        })}
      </List>

      <Box sx={{ flexGrow: 1 }} />

      <AnimatedBox 
        style={useSpring({
          from: { opacity: 0, transform: 'translateY(20px)' },
          to: { opacity: layoutVisible ? 1 : 0, transform: layoutVisible ? 'translateY(0)' : 'translateY(20px)' },
          delay: 800,
          config: { tension: 280, friction: 60 }
        })}
        sx={{ p: 2, textAlign: 'center' }}
      >
        <AnimatedButton
          variant="outlined"
          startIcon={<HelpIcon />}
          fullWidth
          style={useSpring({
            from: { opacity: 0.8, transform: 'scale(0.95)' },
            to: async (next) => {
              await next({ opacity: 1, transform: 'scale(1)' });
              await next({ opacity: 0.98, transform: 'scale(0.99)' });
              await next({ opacity: 1, transform: 'scale(1)' });
            },
            delay: 1000,
            config: { tension: 280, friction: 20 }
          })}
          sx={{ 
            borderRadius: 3, 
            fontWeight: 500,
            position: 'relative',
            overflow: 'hidden',
            transition: 'all 0.3s ease',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
              borderColor: theme.palette.primary.main,
              '&::after': {
                opacity: 1,
                transform: 'translateX(100%)',
              }
            },
            '&::after': {
              content: '""',
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              background: `linear-gradient(90deg, transparent, ${alpha(theme.palette.primary.main, 0.2)}, transparent)`,
              opacity: 0,
              transform: 'translateX(-100%)',
              transition: 'transform 0.6s ease, opacity 0.6s ease',
            }
          }}
        >
          <HumanText>Documentation</HumanText>
        </AnimatedButton>
      </AnimatedBox>
    </>
  );

  const { error, clearError } = useError();

  return (
    <Box sx={{ display: 'flex', height: '100%' }}>
      {/* App Bar */}
      <AnimatedAppBar
        position="fixed"
        style={appBarAnimation}
        sx={{
          width: { md: `calc(100% - ${drawerWidth}px)` },
          ml: { md: `${drawerWidth}px` },
          borderRadius: 0,
          backdropFilter: 'blur(10px)',
          backgroundColor: 'rgba(255, 255, 255, 0.85)',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.05)',
        }}
      >
        <AnimatedToolbar
          style={useSpring({
            from: { opacity: 0 },
            to: { opacity: layoutVisible ? 1 : 0 },
            delay: 100,
            config: { tension: 280, friction: 60 }
          })}
        >
          <AnimatedIconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            style={useSpring({
              transform: mobileOpen ? 'rotate(180deg)' : 'rotate(0deg)',
              config: { tension: 200, friction: 20 }
            })}
            sx={{ 
              mr: 2, 
              display: { md: 'none' }, 
              color: 'text.primary',
              transition: 'background-color 0.3s ease',
              '&:hover': {
                backgroundColor: alpha(theme.palette.primary.main, 0.08),
              }
            }}
          >
            <MenuIcon />
          </AnimatedIconButton>

          <AnimatedTypography 
            variant="h6" 
            component="div" 
            style={useSpring({
              from: { opacity: 0, transform: 'translateY(-10px)' },
              to: { opacity: layoutVisible ? 1 : 0, transform: layoutVisible ? 'translateY(0)' : 'translateY(-10px)' },
              delay: 200,
              config: { tension: 280, friction: 60 }
            })}
            sx={{ 
              flexGrow: 1, 
              fontWeight: 600,
              color: theme.palette.text.primary,
              display: { xs: 'none', sm: 'block' }
            }}
          >
            {navItems.find(item => item.path === location.pathname)?.name || 'Dashboard'}
          </AnimatedTypography>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Tooltip title="Toggle theme">
              <AnimatedIconButton 
                style={themeToggleAnimation}
                sx={{ 
                  color: 'text.secondary',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    color: theme.palette.primary.main,
                    backgroundColor: alpha(theme.palette.primary.main, 0.08),
                    transform: 'rotate(30deg)',
                  }
                }}
              >
                <LightModeIcon />
              </AnimatedIconButton>
            </Tooltip>
            
            <Tooltip title="Notifications">
              <AnimatedIconButton 
                style={bellAnimation}
                sx={{ 
                  color: 'text.secondary',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    color: theme.palette.primary.main,
                    backgroundColor: alpha(theme.palette.primary.main, 0.08),
                    transform: 'translateY(-2px)',
                  },
                  '&:active': {
                    transform: 'translateY(0)',
                  }
                }}
                onClick={() => {
                  // Add pulse animation to bell
                  const bellIcon = document.querySelector('.notification-bell svg');
                  if (bellIcon) {
                    bellIcon.animate([
                      { transform: 'rotate(-10deg)' },
                      { transform: 'rotate(10deg)' },
                      { transform: 'rotate(-10deg)' },
                      { transform: 'rotate(10deg)' },
                      { transform: 'rotate(0deg)' }
                    ], {
                      duration: 500,
                      easing: 'ease-in-out'
                    });
                  }
                }}
                className="notification-bell"
              >
                <NotificationsIcon />
              </AnimatedIconButton>
            </Tooltip>
            
            <Tooltip title="Account">
              <AnimatedIconButton
                onClick={handleUserMenuOpen}
                style={avatarAnimation}
                sx={{ p: 0.5 }}
              >
                <AnimatedAvatar
                  style={useSpring({
                    from: { transform: 'scale(0.9)' },
                    to: { transform: layoutVisible ? 'scale(1)' : 'scale(0.9)' },
                    config: { tension: 200, friction: 20 }
                  })}
                  sx={{
                    width: 40,
                    height: 40,
                    border: `2px solid ${theme.palette.primary.main}`,
                    bgcolor: theme.palette.primary.light,
                    transition: 'all 0.3s ease',
                    boxShadow: '0 2px 10px rgba(0, 0, 0, 0.1)',
                    '&:hover': {
                      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
                      transform: 'scale(1.05)',
                    }
                  }}
                >
                  <PersonIcon />
                </AnimatedAvatar>
              </AnimatedIconButton>
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
                  overflow: 'hidden',
                  animation: 'slideIn 0.25s ease-out',
                  '@keyframes slideIn': {
                    from: { opacity: 0, transform: 'translateY(-10px)' },
                    to: { opacity: 1, transform: 'translateY(0)' }
                  },
                }
              }}
              transformOrigin={{ horizontal: 'right', vertical: 'top' }}
              anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
            >
              <MenuItem onClick={handleUserMenuClose} sx={{
                transition: 'background-color 0.2s ease',
                '&:hover': {
                  backgroundColor: alpha(theme.palette.primary.main, 0.08),
                }
              }}>
                <ListItemIcon>
                  <PersonIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText primary={<HumanText>Profile</HumanText>} />
              </MenuItem>
              <MenuItem onClick={handleUserMenuClose} sx={{
                transition: 'background-color 0.2s ease',
                '&:hover': {
                  backgroundColor: alpha(theme.palette.primary.main, 0.08),
                }
              }}>
                <ListItemIcon>
                  <SettingsIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText primary={<HumanText>Account Settings</HumanText>} />
              </MenuItem>
              <Divider />
              <MenuItem onClick={handleUserMenuClose} sx={{
                transition: 'background-color 0.2s ease',
                '&:hover': {
                  backgroundColor: alpha(theme.palette.primary.main, 0.08),
                }
              }}>
                <ListItemIcon>
                  <LogoutIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText primary={<HumanText>Logout</HumanText>} />
              </MenuItem>
            </Menu>
          </Box>
        </AnimatedToolbar>
      </AnimatedAppBar>

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
              animation: 'slideIn 0.3s ease-out',
              '@keyframes slideIn': {
                from: { transform: 'translateX(-100%)' },
                to: { transform: 'translateX(0)' }
              }
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
              backgroundImage: 'linear-gradient(rgba(255, 255, 255, 0.95), rgba(245, 247, 250, 0.95))',
              boxShadow: '2px 0 12px rgba(0, 0, 0, 0.03)',
              animation: 'fadeIn 0.5s ease-out',
              '@keyframes fadeIn': {
                from: { opacity: 0 },
                to: { opacity: 1 }
              }
            },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>

      {/* Main content with enhanced white space and material consistency */}
      <AnimatedBox
        component="main"
        style={contentAnimation}
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
        <AnimatedBox 
          style={useSpring({
            from: { opacity: 0 },
            to: { opacity: layoutVisible ? 1 : 0 },
            delay: 300,
            config: { tension: 280, friction: 60 }
          })}
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
            <AnimatedBox 
              style={useSpring({
                from: { opacity: 0, transform: 'translateY(-10px)' },
                to: { opacity: 1, transform: 'translateY(0)' },
                config: { tension: 280, friction: 60 }
              })}
              sx={{ 
                mb: { xs: 3, sm: 4, md: 5 }, 
                maxWidth: 1200, 
                mx: 'auto',
                position: 'relative',
                zIndex: 1,
              }}
            >
              <ErrorHandler error={error} onClose={clearError} />
            </AnimatedBox>
          )}
          <AnimatedBox 
            style={useSpring({
              from: { opacity: 0, transform: 'translateY(20px)' },
              to: { opacity: layoutVisible ? 1 : 0, transform: layoutVisible ? 'translateY(0)' : 'translateY(20px)' },
              delay: 400,
              config: { tension: 280, friction: 60 }
            })}
            sx={{ 
              maxWidth: 1200, 
              mx: 'auto',
              position: 'relative',
              zIndex: 1,
              // Add additional breathing room at the bottom
              pb: { xs: 2, sm: 3, md: 4 },
            }}
          >
            <Outlet />
          </AnimatedBox>
        </AnimatedBox>
      </AnimatedBox>
    </Box>
  );
};

export default Layout;