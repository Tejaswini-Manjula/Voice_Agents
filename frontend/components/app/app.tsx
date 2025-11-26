'use client';

import {
  LiveKitRoom,
  RoomAudioRenderer,
  StartAudio,
} from '@livekit/components-react';

import { SessionProvider } from '@/components/app/session-provider';
import { ViewController } from '@/components/app/view-controller';
import { Toaster } from '@/components/livekit/toaster';
import { useLiveKitConnect } from '@/hooks/useLiveKitConnect';
import type { AppConfig } from '@/app-config';

interface AppProps {
  appConfig: AppConfig;
}

export function App({ appConfig }: AppProps) {
  // ✅ Ensure wsUrl and token are always strings
  const wsUrl = appConfig.wsUrl ?? "";
  const token = appConfig.token ?? "";

  // 🔥 Connect to LiveKit + publish mic
  useLiveKitConnect(wsUrl, token);

  return (
    <LiveKitRoom
      token={token}
      serverUrl={wsUrl}
       audio={true}
      video={false}
      connect={false}     // 👈 We control connection manually using the hook
    >
      <SessionProvider appConfig={appConfig}>
        <main className="grid h-svh grid-cols-1 place-content-center">
          <ViewController />
        </main>
        <StartAudio label="Start Audio" />
        <RoomAudioRenderer />
        <Toaster />
      </SessionProvider>
    </LiveKitRoom>
  );
}

