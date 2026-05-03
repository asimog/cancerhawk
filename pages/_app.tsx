import type { AppProps } from 'next/app';
import Head from 'next/head';
import { GlobalMusicButton } from '@/components/global-music-button';
import { MusicProvider } from '@/components/music-provider';
import { SiteBackground } from '@/components/site-background';
import { ErrorBoundary } from '@/components/error-boundary';
import '@/styles/globals.css';

export default function App({ Component, pageProps }: AppProps) {
  return (
    <MusicProvider>
      <Head>
        <title>CancerHawk</title>
        <meta name="description" content="Autonomous oncology research blocks from CancerHawk." />
        <link rel="icon" href="/icon.png" />
        <link rel="apple-touch-icon" href="/icon.png" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400;1,500;1,600;1,700;1,800&family=Fragment+Mono:ital,wght@0,400;1,400&display=swap"
          rel="stylesheet"
        />
      </Head>
      <SiteBackground />
      <main className="app-shell">
        <ErrorBoundary>
          <Component {...pageProps} />
        </ErrorBoundary>
      </main>
      <GlobalMusicButton />
    </MusicProvider>
  );
}
