/**
 * Sound effects utility for subtle audio feedback
 * Inspired by Apple's iPadOS sound design - minimal, high quality, and never intrusive
 */

// Track user interaction to ensure sounds only play after user interaction
let hasUserInteracted = false;
document.addEventListener('click', () => {
  hasUserInteracted = true;
});

// Volume levels
const VOLUME = {
  VERY_LOW: 0.05,  // Almost imperceptible, for very subtle feedback
  LOW: 0.1,        // Subtle, for common interactions
  MEDIUM: 0.2,     // More noticeable, for confirmations
  HIGH: 0.3        // Most prominent, for important actions
};

// Paths to sound assets
const SOUNDS = {
  TAP: '/assets/sounds/tap.mp3',
  SWITCH: '/assets/sounds/switch.mp3',
  SUCCESS: '/assets/sounds/success.mp3',
  ERROR: '/assets/sounds/error.mp3',
  TRANSITION: '/assets/sounds/transition.mp3',
  HOVER: '/assets/sounds/hover.mp3'
};

// Create and cache audio elements
const audioCache: Record<string, HTMLAudioElement> = {};

/**
 * Play a sound effect with specified volume
 * @param sound The sound file path
 * @param volume Volume level (0-1)
 * @returns Promise that resolves when sound starts playing, or rejects if it fails
 */
export const playSound = async (sound: string, volume = VOLUME.LOW): Promise<void> => {
  // Don't play sounds if user hasn't interacted with the page
  if (!hasUserInteracted) return;
  
  try {
    // Create or reuse audio element
    if (!audioCache[sound]) {
      audioCache[sound] = new Audio(sound);
    }
    
    const audio = audioCache[sound];
    audio.volume = volume;
    audio.currentTime = 0;
    
    return audio.play();
  } catch (error) {
    console.debug('Failed to play sound:', error);
    return Promise.resolve();
  }
};

/**
 * Sound effects for different interactions
 */
export const soundEffects = {
  /**
   * Play a tap/click sound (for buttons and interactive elements)
   */
  tap: () => playSound(SOUNDS.TAP, VOLUME.LOW),
  
  /**
   * Play a switch sound (for toggles, switches)
   */
  switch: () => playSound(SOUNDS.SWITCH, VOLUME.LOW),
  
  /**
   * Play a success sound (for confirmations)
   */
  success: () => playSound(SOUNDS.SUCCESS, VOLUME.MEDIUM),
  
  /**
   * Play an error sound (for errors, warnings)
   */
  error: () => playSound(SOUNDS.ERROR, VOLUME.MEDIUM),
  
  /**
   * Play a transition sound (for page changes)
   */
  transition: () => playSound(SOUNDS.TRANSITION, VOLUME.VERY_LOW),
  
  /**
   * Play a hover sound (for hover states)
   */
  hover: () => playSound(SOUNDS.HOVER, VOLUME.VERY_LOW)
};

/**
 * Enable or disable all sound effects
 * @param enabled Whether sound effects should be enabled
 */
export const setSoundEffectsEnabled = (enabled: boolean): void => {
  hasUserInteracted = enabled;
};

export default soundEffects;