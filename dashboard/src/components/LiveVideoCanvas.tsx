import { useEffect, useRef } from "react";

import { useWebSocket } from "../hooks/useWebSocket";

interface LiveVideoCanvasProps {
  cameraId: string;
  quality?: number;
  maxFps?: number;
}

export function LiveVideoCanvas({ cameraId, quality = 85, maxFps = 30 }: LiveVideoCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  const wsUrl = `${protocol}//${host}/ws/stream/${cameraId}?quality=${quality}&max_fps=${maxFps}&mode=binary`;

  const handleMessage = async (data: ArrayBuffer) => {
    try {
      const blob = new Blob([data], { type: "image/jpeg" });
      const bitmap = await createImageBitmap(blob);
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const ctx = ctxRef.current;
      if (!ctx) return;
      ctx.drawImage(bitmap, 0, 0);
      bitmap.close();
    } catch {
      // skip unrenderable frames
    }
  };

  useEffect(() => {
    ctxRef.current = canvasRef.current?.getContext("2d") ?? null;
  }, []);

  const { connected } = useWebSocket(wsUrl, { onMessage: handleMessage });

  return (
    <div className="relative bg-black rounded-lg overflow-hidden">
      <canvas
        ref={canvasRef}
        className="w-full h-auto max-h-[70vh] object-contain"
      />
      <div className="absolute top-2 left-2 flex items-center gap-2">
        <span
          className={`w-2 h-2 rounded-full ${connected ? "bg-green-500" : "bg-red-500"}`}
        />
        <span className="text-xs text-white/70 font-mono">{cameraId}</span>
      </div>
      {!connected && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/60">
          <p className="text-gray-400 text-sm">Connecting to stream...</p>
        </div>
      )}
    </div>
  );
}
