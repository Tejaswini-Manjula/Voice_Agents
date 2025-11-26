import { useEffect } from "react";
import { Room } from "livekit-client";

export function useLiveKitConnect(wsUrl: string, token: string) {
  useEffect(() => {
    if (!wsUrl || !token) return;

    const room = new Room();

    room.connect(wsUrl, token)
      .then(() => {
        console.log("Connected to LiveKit room");
      })
      .catch((err) => {
        console.error("Failed to connect:", err);
      });

    return () => {
      room.disconnect();
    };
  }, [wsUrl, token]);
}
