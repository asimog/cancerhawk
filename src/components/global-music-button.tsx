'use client';

import { useMusic } from '@/components/music-provider';

export function GlobalMusicButton() {
  const music = useMusic();
  return (
    <button
      aria-label={music.isPlaying ? 'Pause music' : 'Play music'}
      className={`global-play-pause-btn ${music.isPlaying ? 'is-playing' : ''}`}
      onClick={() => void music.toggle()}
      title={music.status}
      type="button"
    >
      {music.isPlaying ? 'Ⅱ' : '▶'}
    </button>
  );
}
