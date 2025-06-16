import React, { createContext, useContext, useState, useEffect } from 'react';
import { setSoundEffectsEnabled } from '../utils/soundEffects';

interface AnimationContextType {
  /** Whether animations are enabled */
  animationsEnabled: boolean;
  /** Whether sound effects are enabled */
  soundEffectsEnabled: boolean;
  /** Function to enable animations */
  enableAnimations: () => void;
  /** Function to disable animations */
  disableAnimations: () => void;
  /** Function to toggle animations */
  toggleAnimations: () => void;
  /** Function to enable sound effects */
  enableSoundEffects: () => void;
  /** Function to disable sound effects */
  disableSoundEffects: () => void;
  /** Function to toggle sound effects */
  toggleSoundEffects: () => void;
}

// Create the context with default values
const AnimationContext = createContext<AnimationContextType>({
  animationsEnabled: true,
  soundEffectsEnabled: true,
  enableAnimations: () => {},
  disableAnimations: () => {},
  toggleAnimations: () => {},
  enableSoundEffects: () => {},
  disableSoundEffects: () => {},
  toggleSoundEffects: () => {},
});

/**
 * A hook to access the animation context
 * @returns Animation context
 */
export const useAnimationContext = () => useContext(AnimationContext);

/**
 * Storage keys for saving preferences
 */
const ANIMATION_PREFERENCE_KEY = 'langchain-hana-animations-enabled';
const SOUND_PREFERENCE_KEY = 'langchain-hana-sounds-enabled';

/**
 * Props for the AnimationProvider component
 */
interface AnimationProviderProps {
  children: React.ReactNode;
  /** Default animation enabled state (defaults to true) */
  defaultAnimationsEnabled?: boolean;
  /** Default sound effects enabled state (defaults to true) */
  defaultSoundEffectsEnabled?: boolean;
}

/**
 * Provider component for animation and sound settings
 */
export const AnimationProvider: React.FC<AnimationProviderProps> = ({
  children,
  defaultAnimationsEnabled = true,
  defaultSoundEffectsEnabled = true,
}) => {
  // Try to get the stored animation preference, fallback to default
  const getInitialAnimationState = (): boolean => {
    try {
      const savedPreference = localStorage.getItem(ANIMATION_PREFERENCE_KEY);
      // If no preference is saved, use the default
      if (savedPreference === null) return defaultAnimationsEnabled;
      // Otherwise, use the saved preference
      return savedPreference === 'true';
    } catch (error) {
      // If there's an error (e.g., localStorage not available), use the default
      console.warn('Failed to get animation preference from localStorage', error);
      return defaultAnimationsEnabled;
    }
  };
  
  // Try to get the stored sound preference, fallback to default
  const getInitialSoundState = (): boolean => {
    try {
      const savedPreference = localStorage.getItem(SOUND_PREFERENCE_KEY);
      // If no preference is saved, use the default
      if (savedPreference === null) return defaultSoundEffectsEnabled;
      // Otherwise, use the saved preference
      return savedPreference === 'true';
    } catch (error) {
      // If there's an error (e.g., localStorage not available), use the default
      console.warn('Failed to get sound preference from localStorage', error);
      return defaultSoundEffectsEnabled;
    }
  };

  const [animationsEnabled, setAnimationsEnabled] = useState<boolean>(getInitialAnimationState);
  const [soundEffectsEnabled, setSoundEffectsEnabled] = useState<boolean>(getInitialSoundState);

  // Save animation preference to localStorage whenever it changes
  useEffect(() => {
    try {
      localStorage.setItem(ANIMATION_PREFERENCE_KEY, String(animationsEnabled));
    } catch (error) {
      console.warn('Failed to save animation preference to localStorage', error);
    }
  }, [animationsEnabled]);
  
  // Save sound preference to localStorage whenever it changes
  useEffect(() => {
    try {
      localStorage.setItem(SOUND_PREFERENCE_KEY, String(soundEffectsEnabled));
      // Update the sound effects utility
      setSoundEffectsEnabled(soundEffectsEnabled);
    } catch (error) {
      console.warn('Failed to save sound preference to localStorage', error);
    }
  }, [soundEffectsEnabled]);

  // Animation control functions
  const enableAnimations = () => setAnimationsEnabled(true);
  const disableAnimations = () => setAnimationsEnabled(false);
  const toggleAnimations = () => setAnimationsEnabled(prev => !prev);
  
  // Sound effect control functions
  const enableSoundEffects = () => setSoundEffectsEnabled(true);
  const disableSoundEffects = () => setSoundEffectsEnabled(false);
  const toggleSoundEffects = () => setSoundEffectsEnabled(prev => !prev);

  const value = {
    animationsEnabled,
    soundEffectsEnabled,
    enableAnimations,
    disableAnimations,
    toggleAnimations,
    enableSoundEffects,
    disableSoundEffects,
    toggleSoundEffects,
  };

  return (
    <AnimationContext.Provider value={value}>
      {children}
    </AnimationContext.Provider>
  );
};

export default AnimationProvider;