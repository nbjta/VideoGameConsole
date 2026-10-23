// #3 Lobbyen må ikke crashe uden Supabase. Login og leaderboard kræver env-variablerne.
import { createClient, type SupabaseClient } from "@supabase/supabase-js";

const url = import.meta.env.VITE_SUPABASE_URL as string | undefined;
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

const OFFLINE_MESSAGE =
  "Supabase is not configured. Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY in .env.local.";

function offlineSupabase(): SupabaseClient {
  const unavailable = { message: OFFLINE_MESSAGE };
  const query = {
    select() {
      return query;
    },
    order() {
      return query;
    },
    limit() {
      return Promise.resolve({ data: null, error: unavailable });
    },
  };
  const channel = {
    on() {
      return channel;
    },
    subscribe() {
      return channel;
    },
  };
  const client = {
    auth: {
      getSession: async () => ({ data: { session: null }, error: null }),
      onAuthStateChange: () => ({ data: { subscription: { unsubscribe() {} } } }),
      signOut: async () => ({ error: null }),
      signInWithOAuth: async () => ({ data: { provider: "google", url: null }, error: unavailable }),
    },
    from() {
      return query;
    },
    channel() {
      return channel;
    },
    removeChannel: async () => "ok",
  };
  return client as unknown as SupabaseClient;
}

export const supabase: SupabaseClient =
  url && anonKey
    ? createClient(url, anonKey, {
        auth: {
          persistSession: true,
          autoRefreshToken: true,
          detectSessionInUrl: true,
        },
      })
    : offlineSupabase();
