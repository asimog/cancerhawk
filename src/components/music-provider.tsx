'use client';

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import type { MutableRefObject } from 'react';

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

type YouTubeEntry = {
  id: string;
  title: string;
  videoId: string;
};

type YouTubeResolved = {
  kind: 'video' | 'playlist';
  title: string;
  videoId: string | null;
  playlistId: string | null;
  entries: YouTubeEntry[];
  warning: string | null;
};

type YouTubePlayer = {
  loadPlaylist: (playlist: string[] | string, index?: number) => void;
  loadVideoById: (videoId: string) => void;
  playVideo: () => void;
  pauseVideo: () => void;
  nextVideo: () => void;
  previousVideo: () => void;
  getPlayerState: () => number;
  getCurrentTime: () => number;
  getDuration: () => number;
  destroy: () => void;
};

type YouTubeNamespace = {
  Player: new (
    element: Element,
    options: {
      width?: string;
      height?: string;
      videoId?: string;
      playerVars?: Record<string, string | number>;
      events?: {
        onReady?: () => void;
        onStateChange?: (event: { data: number }) => void;
      };
    },
  ) => YouTubePlayer;
};

type MusicContextValue = {
  tracks: Track[];
  selectedTrack: Track;
  sourceKind: 'mp3' | 'youtube';
  youtubeUrl: string;
  youtubeTitle: string;
  youtubeEntries: YouTubeEntry[];
  youtubeLoading: boolean;
  isPlaying: boolean;
  status: string;
  features: AudioFeatures;
  toggle: () => Promise<void>;
  next: () => Promise<void>;
  previous: () => Promise<void>;
  select: (id: string) => Promise<void>;
  setYoutubeUrl: (url: string) => void;
  loadYoutube: () => Promise<void>;
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
const YOUTUBE_PLAYING = 1;

declare global {
  interface Window {
    YT?: YouTubeNamespace;
    onYouTubeIframeAPIReady?: () => void;
    __cancerHawkYoutubeApiPromise?: Promise<YouTubeNamespace>;
  }
}

function resolveTrackUrl(value: string) {
  if (/^https?:\/\//i.test(value) || value.startsWith('blob:')) return value;
  return `/music/${value.replace(/^\/+/, '')}`;
}

function labelFromFile(value: string) {
  return value.split('/').pop()?.replace(/\.[^/.]+$/, '').replace(/[-_]+/g, ' ') || value;
}

function loadYouTubeIframeApi(): Promise<YouTubeNamespace> {
  if (window.YT?.Player) return Promise.resolve(window.YT);
  if (window.__cancerHawkYoutubeApiPromise) return window.__cancerHawkYoutubeApiPromise;

  window.__cancerHawkYoutubeApiPromise = new Promise((resolve, reject) => {
    window.onYouTubeIframeAPIReady = () => {
      if (window.YT?.Player) resolve(window.YT);
      else reject(new Error('YouTube player API loaded without a player.'));
    };
    const existing = document.querySelector<HTMLScriptElement>('script[src="https://www.youtube.com/iframe_api"]');
    if (existing) return;
    const script = document.createElement('script');
    script.src = 'https://www.youtube.com/iframe_api';
    script.async = true;
    script.onerror = () => reject(new Error('YouTube player API could not load.'));
    document.head.appendChild(script);
  });

  return window.__cancerHawkYoutubeApiPromise;
}

function synthYoutubeFeatures(time: number, duration: number, salt: number, beatRef: MutableRefObject<number>): AudioFeatures {
  const progress = duration > 0 ? time / duration : 0;
  const bass = Math.max(0, Math.min(1, 0.34 + Math.sin(time * 2.1 + salt) ** 2 * 0.44));
  const mid = Math.max(0, Math.min(1, 0.22 + Math.sin(time * 1.34 + salt * 2) ** 2 * 0.42));
  const high = Math.max(0, Math.min(1, 0.16 + Math.sin(time * 3.8 + progress * 8 + salt) ** 2 * 0.38));
  const volume = Math.max(0, Math.min(1, 0.22 + bass * 0.34 + mid * 0.22 + high * 0.16));
  const bucket = Math.floor(time * (2.1 + salt * 0.05));
  const beat = bucket !== beatRef.current;
  if (beat) beatRef.current = bucket;
  return { bass, mid, high, volume, beat, isPlaying: true };
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
  const [sourceKind, setSourceKind] = useState<'mp3' | 'youtube'>('mp3');
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [youtubeTitle, setYoutubeTitle] = useState('');
  const [youtubeEntries, setYoutubeEntries] = useState<YouTubeEntry[]>([]);
  const [youtubeLoading, setYoutubeLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [status, setStatus] = useState('Ready.');
  const [features, setFeatures] = useState<AudioFeatures>(emptyFeatures);
  const youtubeHostRef = useRef<HTMLDivElement | null>(null);
  const youtubePlayerRef = useRef<YouTubePlayer | null>(null);
  const youtubePollRef = useRef(0);
  const youtubeBeatRef = useRef(0);

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
      if (sourceKind === 'mp3') setFeatures({ bass, mid, high, volume, beat, isPlaying: Boolean(audioRef.current && !audioRef.current.paused) });
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
    youtubePlayerRef.current?.pauseVideo();
    window.clearInterval(youtubePollRef.current);
    setSourceKind('mp3');
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
    if (sourceKind === 'youtube' && youtubePlayerRef.current) {
      if (isPlaying) {
        youtubePlayerRef.current.pauseVideo();
        setIsPlaying(false);
        setStatus('Paused YouTube.');
      } else {
        youtubePlayerRef.current.playVideo();
        setIsPlaying(true);
        setStatus(`Playing ${youtubeTitle || 'YouTube'}`);
      }
      return;
    }
    const audio = await ensureAudio();
    if (!audio.paused) {
      audio.pause();
      setStatus('Paused.');
      return;
    }
    await play(selectedTrack);
  }, [ensureAudio, play, selectedTrack]);

  const next = useCallback(async () => {
    if (sourceKind === 'youtube' && youtubePlayerRef.current) {
      youtubePlayerRef.current.nextVideo();
      setIsPlaying(true);
      return;
    }
    const index = tracks.findIndex((track) => track.id === selectedTrack.id);
    await select(tracks[(index + 1) % tracks.length].id);
  }, [select, selectedTrack.id, sourceKind, tracks]);

  const previous = useCallback(async () => {
    if (sourceKind === 'youtube' && youtubePlayerRef.current) {
      youtubePlayerRef.current.previousVideo();
      setIsPlaying(true);
      return;
    }
    const index = tracks.findIndex((track) => track.id === selectedTrack.id);
    await select(tracks[(index - 1 + tracks.length) % tracks.length].id);
  }, [select, selectedTrack.id, sourceKind, tracks]);

  const startYoutubeFeaturePoll = useCallback((salt: number) => {
    window.clearInterval(youtubePollRef.current);
    youtubePollRef.current = window.setInterval(() => {
      const player = youtubePlayerRef.current;
      if (!player || player.getPlayerState() !== YOUTUBE_PLAYING) {
        setFeatures((current) => ({ ...current, isPlaying: false, beat: false }));
        setIsPlaying(false);
        return;
      }
      setSourceKind('youtube');
      setIsPlaying(true);
      setFeatures(synthYoutubeFeatures(player.getCurrentTime() || 0, player.getDuration() || 1, salt, youtubeBeatRef));
    }, 120);
  }, []);

  const ensureYoutubePlayer = useCallback(async () => {
    if (youtubePlayerRef.current) return youtubePlayerRef.current;
    if (!youtubeHostRef.current) throw new Error('YouTube player is not mounted yet.');
    const api = await loadYouTubeIframeApi();
    const player = await new Promise<YouTubePlayer>((resolve) => {
      const created = new api.Player(youtubeHostRef.current!, {
        width: '1',
        height: '1',
        playerVars: {
          autoplay: 1,
          controls: 0,
          disablekb: 1,
          fs: 0,
          modestbranding: 1,
          playsinline: 1,
          rel: 0,
        },
        events: {
          onReady: () => resolve(created),
          onStateChange: (event) => {
            setIsPlaying(event.data === YOUTUBE_PLAYING);
          },
        },
      });
    });
    youtubePlayerRef.current = player;
    return player;
  }, []);

  const loadYoutube = useCallback(async () => {
    const url = youtubeUrl.trim();
    if (!url) return;
    setYoutubeLoading(true);
    setStatus('Loading YouTube...');
    try {
      const response = await fetch('/api/youtube/resolve', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ url }),
      });
      const media = (await response.json()) as YouTubeResolved & { error?: string };
      if (!response.ok) throw new Error(media.error || 'YouTube could not be loaded.');
      const player = await ensureYoutubePlayer();
      audioRef.current?.pause();
      setSourceKind('youtube');
      setYoutubeTitle(media.title || 'YouTube audio');
      setYoutubeEntries(media.entries || []);
      if (media.entries?.length) player.loadPlaylist(media.entries.map((entry) => entry.videoId), 0);
      else if (media.videoId) player.loadVideoById(media.videoId);
      else if (media.playlistId) player.loadPlaylist(media.playlistId);
      player.playVideo();
      startYoutubeFeaturePoll(Math.max(1, (media.entries || []).length));
      setIsPlaying(true);
      setStatus(media.warning || `Playing ${media.title || 'YouTube audio'}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'YouTube could not be loaded.');
    } finally {
      setYoutubeLoading(false);
    }
  }, [ensureYoutubePlayer, startYoutubeFeaturePoll, youtubeUrl]);

  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafRef.current);
      window.clearInterval(youtubePollRef.current);
      youtubePlayerRef.current?.destroy();
      void contextRef.current?.close();
    };
  }, []);

  return (
    <MusicContext.Provider value={{ tracks, selectedTrack, sourceKind, youtubeUrl, youtubeTitle, youtubeEntries, youtubeLoading, isPlaying, status, features, toggle, next, previous, select, setYoutubeUrl, loadYoutube }}>
      {children}
      <div style={{ height: 1, left: -9999, opacity: 0, overflow: 'hidden', position: 'fixed', top: -9999, width: 1 }} ref={youtubeHostRef} />
    </MusicContext.Provider>
  );
}

export function useMusic() {
  const context = useContext(MusicContext);
  if (!context) throw new Error('useMusic must be used inside MusicProvider');
  return context;
}
