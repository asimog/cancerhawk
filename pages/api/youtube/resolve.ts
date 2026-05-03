import type { NextApiRequest, NextApiResponse } from 'next';
import { isYouTubeUrl, resolveYoutubeMedia } from '@/lib/youtube';

const rateLimit = new Map<string, { count: number; resetAt: number }>();
const WINDOW_MS = 60_000;
const MAX_REQUESTS = 10;

function getClientIp(req: NextApiRequest): string {
  const forwarded = req.headers['x-forwarded-for'];
  if (typeof forwarded === 'string') return forwarded.split(',')[0].trim();
  return req.socket.remoteAddress || 'unknown';
}

function isRateLimited(ip: string): boolean {
  const now = Date.now();
  const record = rateLimit.get(ip);
  if (!record || now > record.resetAt) {
    rateLimit.set(ip, { count: 1, resetAt: now + WINDOW_MS });
    return false;
  }
  if (record.count >= MAX_REQUESTS) return true;
  record.count += 1;
  return false;
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    res.setHeader('allow', 'POST');
    return res.status(405).json({ error: 'Method not allowed.' });
  }

  const ip = getClientIp(req);
  if (isRateLimited(ip)) {
    return res.status(429).json({ error: 'Too many requests. Please slow down.' });
  }

  const url = typeof req.body?.url === 'string' ? req.body.url.trim() : '';
  if (!url || url.length > 2048 || !isYouTubeUrl(url)) {
    return res.status(400).json({ error: 'Paste a valid YouTube video or playlist URL.' });
  }

  // Extra safety: reject URLs that look like they contain command flags
  if (/[\s&;|]$|^-/m.test(url)) {
    return res.status(400).json({ error: 'Invalid URL format.' });
  }

  try {
    const media = await resolveYoutubeMedia(url);
    return res.status(200).json(media);
  } catch (error) {
    return res.status(502).json({
      error: error instanceof Error ? error.message : 'YouTube media could not be resolved right now.',
    });
  }
}
