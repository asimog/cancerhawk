export type BlockMeta = {
  block: number;
  title: string;
  research_goal: string;
  timestamp: string;
  market_price: number;
  section_count: number;
  has_peer_review: boolean;
  has_simulations: boolean;
};

export type Simulation = {
  id: string;
  title: string;
  description: string;
  rationale: string;
  expected_metrics: string[];
};

export type Analysis = {
  market_price: number;
  headline_catalysts?: string[];
  peer_reviews?: Array<Record<string, unknown>>;
  simulations?: Simulation[];
  derived_topics?: Array<Record<string, unknown>>;
  archetypes?: Array<Record<string, unknown>>;
};

export type BlockBundle = {
  number: number;
  meta: BlockMeta;
  analysis: Analysis;
  paper: string;
};
