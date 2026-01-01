import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { User, AppSettings } from '@/types';

interface AppState {
  user: User | null;
  settings: AppSettings;
  setUser: (user: User | null) => void;
  toggleProMode: () => void;
  toggleCinematicMode: () => void;
  updateSettings: (settings: Partial<AppSettings>) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      user: null,
      settings: {
        proMode: false,
        cinematicMode: true,
        theme: 'dark',
      },
      setUser: (user) => set({ user }),
      toggleProMode: () =>
        set((state) => ({
          settings: { ...state.settings, proMode: !state.settings.proMode },
        })),
      toggleCinematicMode: () =>
        set((state) => ({
          settings: {
            ...state.settings,
            cinematicMode: !state.settings.cinematicMode,
          },
        })),
      updateSettings: (newSettings) =>
        set((state) => ({
          settings: { ...state.settings, ...newSettings },
        })),
    }),
    {
      name: 'ai-trader-storage',
    }
  )
);
