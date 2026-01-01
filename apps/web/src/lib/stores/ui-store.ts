'use client';

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UIState {
  proMode: boolean;
  cinematicMode: boolean;
  leftNavCollapsed: boolean;
  rightRailVisible: boolean;
  bottomRailVisible: boolean;
  commandPaletteOpen: boolean;

  toggleProMode: () => void;
  toggleCinematicMode: () => void;
  toggleLeftNav: () => void;
  toggleRightRail: () => void;
  toggleBottomRail: () => void;
  openCommandPalette: () => void;
  closeCommandPalette: () => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      proMode: false,
      cinematicMode: false,
      leftNavCollapsed: false,
      rightRailVisible: true,
      bottomRailVisible: true,
      commandPaletteOpen: false,

      toggleProMode: () =>
        set((state) => ({
          proMode: !state.proMode,
          cinematicMode: false, // Disable cinematic when enabling pro
        })),

      toggleCinematicMode: () =>
        set((state) => ({
          cinematicMode: !state.cinematicMode,
          proMode: false, // Disable pro when enabling cinematic
        })),

      toggleLeftNav: () =>
        set((state) => ({ leftNavCollapsed: !state.leftNavCollapsed })),

      toggleRightRail: () =>
        set((state) => ({ rightRailVisible: !state.rightRailVisible })),

      toggleBottomRail: () =>
        set((state) => ({ bottomRailVisible: !state.bottomRailVisible })),

      openCommandPalette: () => set({ commandPaletteOpen: true }),

      closeCommandPalette: () => set({ commandPaletteOpen: false }),
    }),
    {
      name: 'autopilot-ui-state',
    }
  )
);
