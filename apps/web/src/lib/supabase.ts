import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Helper to get current user
export async function getCurrentUser() {
  const {
    data: { user },
  } = await supabase.auth.getUser();
  return user;
}

// Helper to get user tier
export async function getUserTier() {
  const user = await getCurrentUser();
  if (!user) return null;

  const { data } = await supabase
    .from('profiles')
    .select('tier')
    .eq('user_id', user.id)
    .single();

  return data?.tier || 'basic';
}

// Helper to check tier access
export function canAccessFeature(
  userTier: string,
  requiredTier: 'basic' | 'pro' | 'elite'
): boolean {
  const tierLevels = { basic: 1, pro: 2, elite: 3 };
  return tierLevels[userTier as keyof typeof tierLevels] >= tierLevels[requiredTier];
}
