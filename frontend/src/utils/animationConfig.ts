import { setSoundPaths } from '@finsightdev/ui-animations';

/**
 * Configure the animation system
 * Call this function during application initialization
 */
export function configureAnimationSystem() {
  // Set up sound asset paths
  setSoundPaths({
    TAP: '/assets/sounds/tap.mp3',
    SWITCH: '/assets/sounds/switch.mp3',
    SUCCESS: '/assets/sounds/success.mp3',
    ERROR: '/assets/sounds/error.mp3',
    TRANSITION: '/assets/sounds/transition.mp3',
    HOVER: '/assets/sounds/hover.mp3'
  });
}

export default configureAnimationSystem;