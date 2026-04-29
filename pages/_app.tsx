import type { AppProps } from 'next/app';
import { GlobalMusicButton } from '@/components/global-music-button';
import { MusicProvider } from '@/components/music-provider';
import { SiteBackground } from '@/components/site-background';
import '@/styles/globals.css';

export default function App({ Component, pageProps }: AppProps) {
  return (
    <MusicProvider>
      <SiteBackground />
      <main className="app-shell">
        <Component {...pageProps} />
      </main>
      <GlobalMusicButton />
    </MusicProvider>
  );
}
