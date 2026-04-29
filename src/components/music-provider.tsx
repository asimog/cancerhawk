'use client';

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';

export type AudioFeatures = {
  bass: number;
  mid: number;
  high: number;
  volume: number;
  beat: boolean;
  isPlaying: boolean;
};

type Track = {
  id: string;
  label: string;
  url: string;
};

type MusicContextValue = {
  tracks: Track[];
  selectedTrack: Track;
  isPlaying: boolean;
  status: string;
  features: AudioFeatures;
  toggle: () => Promise<void>;
  next: () => Promise<void>;
  previous: () => Promise<void>;
  select: (id: string) => Promise<void>;
};

const defaultTrack: Track = {
  id: 'default',
  label: '42069.mp3',
  url: '/music/42069.mp3',
};

const emptyFeatures: AudioFeatures = {
  bass: 0,
  mid: 0,
  high: 0,
  volume: 0,
  beat: false,
  isPlaying: false,
};

const MusicContext = createContext<MusicContextValue | null>(null);

function resolveTrackUrl(value: string) {
  if (/^https?:\/\//i.test(value) || value.startsWith('blob:')) return value;
  return `/music/${value.replace(/^\/+/, '')}`;
}

function labelFromFile(value: string) {
  return value.split('/').pop()?.replace(/\.[^/.]+$/, '').replace(/[-_]+/g, ' ') || value;
}

export function MusicProvider({ children }: { children: React.ReactNode }) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const rafRef = useRef(0);
  const beatRef = useRef(0);
  const [tracks, setTracks] = useState<Track[]>([defaultTrack]);
  const [selectedId, setSelectedId] = useState(defaultTrack.id);
  const [isPlaying, setIsPlaying] = useState(false);
  const [status, setStatus] = useState('Ready.');
  const [features, setFeatures] = useState<AudioFeatures>(emptyFeatures);

  const selectedTrack = useMemo(
    () => tracks.find((track) => track.id === selectedId) || tracks[0],
    [selectedId, tracks],
  );

  useEffect(() => {
    let cancelled = false;
    async function loadManifest() {
      try {
        const res = await fetch('/music/playlist.json', { cache: 'no-store' });
        if (!res.ok) return;
        const json = (await res.json()) as { tracks?: Array<string | { file?: string; url?: string; title?: string }> };
        const parsed = (json.tracks || []).map((entry, index): Track | null => {
          const source = typeof entry === 'string' ? entry : entry.url || entry.file || '';
          if (!source) return null;
          return {
            id: `track-${index}`,
            label: typeof entry === 'string' ? labelFromFile(source) : entry.title || labelFromFile(source),
            url: resolveTrackUrl(source),
          };
        }).filter((track): track is Track => Boolean(track));
        if (!cancelled && parsed.length) setTracks([defaultTrack, ...parsed]);
      } catch {
        // Built-in fallback remains available.
      }
    }
    void loadManifest();
    return () => {
      cancelled = true;
    };
  }, []);

  const startAnalyser = useCallback(() => {
    const tick = () => {
      const analyser = analyserRef.current;
      if (!analyser) {
        rafRef.current = requestAnimationFrame(tick);
        return;
      }
      const data = new Uint8Array(analyser.frequencyBinCount);
      analyser.getByteFrequencyData(data);
      const third = Math.max(1, Math.floor(data.length / 3));
      const avg = (start: number, end: number) => {
        let total = 0;
        for (let i = start; i < end; i += 1) total += data[i] || 0;
        return total / Math.max(1, end - start) / 255;
      };
      const bass = avg(0, third);
      const mid = avg(third, third * 2);
      const high = avg(third * 2, data.length);
      const volume = (bass + mid + high) / 3;
      const bucket = Math.floor(performance.now() / 280);
      const beat = bass > 0.48 && bucket !== beatRef.current;
      if (beat) beatRef.current = bucket;
      setFeatures({ bass, mid, high, volume, beat, isPlaying: Boolean(audioRef.current && !audioRef.current.paused) });
      rafRef.current = requestAnimationFrame(tick);
    };
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(tick);
  }, []);

  const ensureAudio = useCallback(async () => {
    if (!audioRef.current) {
      audioRef.current = new Audio();
      audioRef.current.crossOrigin = 'anonymous';
      audioRef.current.addEventListener('play', () => setIsPlaying(true));
      audioRef.current.addEventListener('pause', () => setIsPlaying(false));
      audioRef.current.addEventListener('ended', () => setIsPlaying(false));
    }
    if (!contextRef.current) {
      const context = new AudioContext();
      const analyser = context.createAnalyser();
      analyser.fftSize = 512;
      sourceRef.current = context.createMediaElementSource(audioRef.current);
      sourceRef.current.connect(analyser);
      analyser.connect(context.destination);
      contextRef.current = context;
      analyserRef.current = analyser;
      startAnalyser();
    }
    if (contextRef.current.state === 'suspended') await contextRef.current.resume();
    return audioRef.current;
  }, [startAnalyser]);

  const play = useCallback(async (track: Track) => {
    const audio = await ensureAudio();
    if (audio.src !== new URL(track.url, window.location.href).href) audio.src = track.url;
    await audio.play();
    setStatus(`Playing ${track.label}`);
  }, [ensureAudio]);

  const select = useCallback(async (id: string) => {
    const track = tracks.find((item) => item.id === id) || tracks[0];
    setSelectedId(track.id);
    await play(track);
  }, [play, tracks]);

  const toggle = useCallback(async () => {
    const audio = await ensureAudio();
    if (!audio.paused) {
      audio.pause();
      setStatus('Paused.');
      return;
    }
    await play(selectedTrack);
  }, [ensureAudio, play, selectedTrack]);

  const next = useCallback(async () => {
    const index = tracks.findIndex((track) => track.id === selectedTrack.id);
    await select(tracks[(index + 1) % tracks.length].id);
  }, [select, selectedTrack.id, tracks]);

  const previous = useCallback(async () => {
    const index = tracks.findIndex((track) => track.id === selectedTrack.id);
    await select(tracks[(index - 1 + tracks.length) % tracks.length].id);
  }, [select, selectedTrack.id, tracks]);

  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafRef.current);
      void contextRef.current?.close();
    };
  }, []);

  return (
    <MusicContext.Provider value={{ tracks, selectedTrack, isPlaying, status, features, toggle, next, previous, select }}>
      {children}
    </MusicContext.Provider>
  );
}

export function useMusic() {
  const context = useContext(MusicContext);
  if (!context) throw new Error('useMusic must be used inside MusicProvider');
  return context;
}
