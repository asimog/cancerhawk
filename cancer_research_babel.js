class MerkleTree {
  constructor(leaves) {
    this.leaves = leaves.map(leaf => this.hash(leaf));
    this.tree = [];
    this.buildTree();
  }

  hash(data) {
    // Simple hash function for demo - in production use crypto.subtle
    let hash = 0;
    const str = JSON.stringify(data);
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash).toString(16).padStart(8, '0');
  }

  buildTree() {
    this.tree = [this.leaves];
    let currentLevel = this.leaves;

    while (currentLevel.length > 1) {
      const nextLevel = [];
      for (let i = 0; i < currentLevel.length; i += 2) {
        const left = currentLevel[i];
        const right = i + 1 < currentLevel.length ? currentLevel[i + 1] : left;
        nextLevel.push(this.hash(left + right));
      }
      this.tree.push(nextLevel);
      currentLevel = nextLevel;
    }
  }

  getRoot() {
    return this.tree[this.tree.length - 1][0];
  }

  getProof(index) {
    const proof = [];
    let currentIndex = index;

    for (let level = 0; level < this.tree.length - 1; level++) {
      const isLeft = currentIndex % 2 === 0;
      const siblingIndex = isLeft ? currentIndex + 1 : currentIndex - 1;

      if (siblingIndex < this.tree[level].length) {
        proof.push({
          hash: this.tree[level][siblingIndex],
          isLeftSibling: !isLeft
        });
      }

      currentIndex = Math.floor(currentIndex / 2);
    }

    return proof;
  }

  verifyProof(leaf, proof, root) {
    let hash = this.hash(leaf);

    for (const { hash: siblingHash, isLeftSibling } of proof) {
      if (isLeftSibling) {
        hash = this.hash(siblingHash + hash);
      } else {
        hash = this.hash(hash + siblingHash);
      }
    }

    return hash === root;
  }
}

class CancerResearchBabel {
  constructor() {
    this.seedCode = "e676098e48313c65989af66900ba43461e738168c3bab7ffbb12e0c96319ed9b6a0e6f5fe1f1737291148c25db2a565b35cf6fe9b00d997c8c9879e735fa81d8SN_3a7c39f14e0c478bc1b8b33ee4e7b4d18e2c8659e7a37f2f5b1b464c8b6f5b19RA_93afb3fdc190a2c9450f42a5e55474db9f69f3630cbd33b783af33c23093742f";
    this.agents = [];
    this.researchBlocks = [];
    this.currentBlock = null;
    this.agentMerkleRoots = new Map(); // Track agent state Merkle roots
    this.blockInterval = 1 * 60 * 1000; // 1 minute for testing, change to 10 * 60 * 1000 for production
    this.minAgents = 42;
    this.agentsPerBlock = 10; // 10 agents evaluate each block
    this.providers = ['openai/gpt-4o', 'anthropic/claude-3.5-sonnet', 'google/gemini-2.0', 'meta/llama-3.1-70b', 'mistral/mistral-7b'];
    this.loadData();
    this.initializeAgents();
    this.displayCurrentBlock();
    this.displayLeaderboard();
    this.displayRecentBlocks();
    this.startSimulation();
  }

  loadData() {
    const data = localStorage.getItem('cancerResearchData');
    if (data) {
      const parsed = JSON.parse(data);
      this.agents = parsed.agents || [];
      this.researchBlocks = parsed.blocks || [];
      this.currentBlock = parsed.currentBlock;
    }
  }

  saveData() {
    const data = {
      agents: this.agents,
      blocks: this.researchBlocks,
      currentBlock: this.currentBlock
    };
    localStorage.setItem('cancerResearchData', JSON.stringify(data));
  }

  initializeAgents() {
    const specialties = [
      'Molecular Oncology', 'Immunotherapy', 'Gene Therapy', 'Clinical Trials',
      'Translational Research', 'Cancer Genomics', 'Drug Development',
      'Radiation Oncology', 'Surgical Oncology', 'Pathology', 'Radiology',
      'Bioinformatics', 'Systems Biology', 'Biostatistics', 'Epidemiology',
      'Tumor Biology', 'Metastasis Research', 'Stem Cell Therapy',
      'Nanotechnology', 'Viral Oncology', 'Endocrine Oncology',
      'Hematologic Oncology', 'Pediatric Oncology', 'Geriatric Oncology',
      'Precision Medicine', 'Liquid Biopsies', 'AI in Oncology',
      'Pharmacokinetics', 'Toxicology', 'Regulatory Affairs',
      'Health Economics', 'Patient Outcomes', 'Palliative Care',
      'Cancer Prevention', 'Screening Methods', 'Biomarker Discovery',
      'Targeted Therapies', 'Chemotherapy', 'Hormone Therapy',
      'Cell Signaling', 'Apoptosis Research', 'Angiogenesis',
      'Microbiome Research', 'Nutrition Oncology', 'Exercise Oncology'
    ];

    const institutions = [
      'Memorial Sloan Kettering', 'MD Anderson', 'Dana-Farber', 'Mayo Clinic',
      'Johns Hopkins', 'UCLA Medical Center', 'Stanford Medicine',
      'Massachusetts General Hospital', 'Cleveland Clinic', 'Mount Sinai',
      'University of Pennsylvania', 'Duke University Hospital', 'Vanderbilt',
      'University of Michigan', 'Northwestern Medicine', 'UCSF Medical Center',
      'Baylor Scott & White', 'Cedars-Sinai', 'NYU Langone', 'Rush University',
      'Georgetown University Hospital', 'George Washington University Hospital',
      'Inova Health System', 'MedStar Health', 'Sibley Memorial Hospital',
      'Suburban Hospital', 'Holy Cross Hospital', 'Adventist Healthcare',
      'Anne Arundel Medical Center', 'Atlantic General Hospital', 'CalvertHealth',
      'Caroline County Health Department', 'Charles County Department of Health',
      'Chester River Health System', 'Dorchester County Health Department',
      'Eastern Shore Hospital Center', 'Garrett County Health Department',
      'Greater Baltimore Medical Center', 'Harford County Health Department',
      'Howard County General Hospital', 'Kent County Health Department',
      'Mercy Medical Center', 'Montgomery General Hospital', 'Peninsula Regional',
      'Prince George\'s Hospital Center', 'Shady Grove Adventist Hospital',
      'Sinai Hospital of Baltimore', 'St. Agnes Hospital', 'St. Joseph Medical Center',
      'St. Mary\'s Hospital', 'Union Memorial Hospital', 'University of Maryland Medical Center',
      'Upper Chesapeake Medical Center', 'Washington Adventist Hospital',
      'Western Maryland Health System', 'White Oak Medical Center'
    ];

    while (this.agents.length < this.minAgents) {
      const specialty = specialties[Math.floor(Math.random() * specialties.length)];
      const institution = institutions[Math.floor(Math.random() * institutions.length)];
      const provider = this.providers[Math.floor(Math.random() * this.providers.length)];

      this.agents.push({
        id: `agent_${this.agents.length + 1}`,
        name: `Dr. ${this.generateName()}`,
        specialty,
        institution,
        provider,
        wallet: { balance: 1000, totalSpent: 0, totalEarned: 0 },
        predictions: [],
        expertise: Math.random() * 0.5 + 0.5, // 0.5-1.0
        reputation: Math.random() * 50 + 50, // 50-100
        contributions: 0
      });
    }
  }

  generateName() {
    const firstNames = ['Alex', 'Jordan', 'Taylor', 'Morgan', 'Casey', 'Riley', 'Avery', 'Quinn', 'Skyler', 'Reese', 'Dakota', 'Sage', 'Rowan', 'Ellis', 'Finley', 'River', 'Emerson', 'Hayden', 'Logan', 'Parker', 'Sawyer', 'Tristan', 'Blake', 'Cameron', 'Devon', 'Ellis', 'Francis', 'Garrett', 'Hunter', 'Ian', 'Jesse', 'Kendall', 'Lee', 'Morgan', 'Noel', 'Owen', 'Pat', 'Quinn', 'Robin', 'Sam', 'Terry', 'Ulysses', 'Val', 'Wynn', 'Xander', 'Yuri', 'Zane'];
    const lastNames = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker', 'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores', 'Green', 'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell', 'Mitchell', 'Carter', 'Roberts'];
    return `${firstNames[Math.floor(Math.random() * firstNames.length)]} ${lastNames[Math.floor(Math.random() * lastNames.length)]}`;
  }

  startSimulation() {
    if (!this.currentBlock) {
      this.issueNewBlock();
    }
    setInterval(() => this.processBlock(), this.blockInterval);
  }

  issueNewBlock() {
    let topics = [];
    if (this.researchBlocks.length === 0) {
      // First block uses the seed code
      topics = [this.seedCode];
    } else {
      // Generate 10 new topics derived from previous winners
      topics = this.generateNewTopics(10);
    }

    this.currentBlock = {
      id: `block_${Date.now()}`,
      topics,
      issuedAt: new Date().toISOString(),
      evaluations: [], // Will be array of arrays (one per topic)
      winners: [], // One winner per topic
      status: 'active'
    };

    this.saveData();
    this.displayCurrentBlock();
  }

  generateNewTopics(count) {
    const topics = [];
    for (let i = 0; i < count; i++) {
      if (this.researchBlocks.length > 0) {
        // Derive from random previous winner
        const randomBlock = this.researchBlocks[Math.floor(Math.random() * this.researchBlocks.length)];
        const winner = randomBlock.winners[Math.floor(Math.random() * randomBlock.winners.length)];
        topics.push(this.generateDerivedTopic(winner.evaluation));
      } else {
        // Fallback initial topics
        topics.push(this.generateInitialTopic());
      }
    }
    return topics;
  }

  generateInitialTopic() {
    const initialTopics = [
      'CRISPR-based universal cancer vaccine targeting shared neoantigens',
      'AI-driven drug discovery for undruggable cancer targets using quantum computing',
      'Microbiome engineering to prevent cancer metastasis through gut-brain-cancer axis',
      'Nanoparticle delivery systems for organ-specific cancer targeting with minimal toxicity',
      'Epigenetic reprogramming to reverse cancer stem cell differentiation',
      'Viral oncolysis combined with CRISPR enhancement for selective tumor destruction',
      'Metabolic reprogramming inhibitors for cancer cell starvation therapy',
      'Immune checkpoint modulators with tissue-specific delivery to overcome resistance',
      'Telomere lengthening inhibitors combined with senolytics for aging-related cancers',
      'RNA-based therapeutics for non-coding RNA dysregulation in cancer'
    ];
    return initialTopics[Math.floor(Math.random() * initialTopics.length)];
  }

  generateDerivedTopic(previousEvaluation) {
    const derivations = [
      `Building on ${previousEvaluation.keyInsight}, explore ${previousEvaluation.followUpQuestion}`,
      `Given ${previousEvaluation.probability} success probability, investigate ${previousEvaluation.riskFactors[0]} mitigation strategies`,
      `Extending ${previousEvaluation.impactScore}/10 impact, develop ${previousEvaluation.nextSteps[0]}`,
      `Addressing ${previousEvaluation.limitations}, propose ${previousEvaluation.alternativeApproaches[0]}`,
      `Combining ${previousEvaluation.methodology} with emerging ${['quantum biology', 'synthetic biology', 'computational oncology', 'organoid technology', 'single-cell sequencing'][Math.floor(Math.random() * 5)]}`
    ];
    return derivations[Math.floor(Math.random() * derivations.length)];
  }

  async processBlock() {
    if (!this.currentBlock || this.currentBlock.status !== 'active') return;

    const blockNumber = this.researchBlocks.length + 1;
    console.log(`🔬 Processing Block #${blockNumber} with ${this.currentBlock.topics.length} topics...`);

    // Select 10 random agents for this block
    const selectedAgents = this.selectRandomAgents(this.agentsPerBlock);
    console.log(`👥 Selected ${selectedAgents.length} agents:`, selectedAgents.map(a => a.name));

    // Each agent evaluates all topics in the block
    console.log('⚡ Running evaluations...');
    this.currentBlock.evaluations = this.currentBlock.topics.map((topic, topicIndex) => {
      console.log(`  Topic ${topicIndex + 1}: ${topic.substring(0, 60)}...`);
      return selectedAgents.map(agent => {
        const evaluation = this.simulateEvaluation(agent, topic);
        console.log(`    ${agent.name}: ${(evaluation.evaluation.probability * 100).toFixed(1)}% prob, ${evaluation.tokensSpent} tokens`);
        return {
          ...evaluation,
          participated: true
        };
      });
    });

    // Determine winner for each topic (highest token spend among evaluators)
    console.log('🏆 Calculating winners...');
    this.currentBlock.winners = this.currentBlock.evaluations.map((topicEvaluations, topicIndex) => {
      const winner = topicEvaluations.reduce((prev, current) =>
        (prev.tokensSpent > current.tokensSpent) ? prev : current
      );
      console.log(`  Topic ${topicIndex + 1} Winner: ${winner.agentName} (${winner.tokensSpent} tokens, ${(winner.evaluation.probability * 100).toFixed(1)}% prob)`);
      return winner;
    });

    // Distribute rewards for each topic
    const totalTokensSpent = this.currentBlock.evaluations.flat().reduce((sum, e) => sum + e.tokensSpent, 0);
    const rewardPool = totalTokensSpent * 0.1;
    const rewardPerWinner = rewardPool / this.currentBlock.winners.length;

    console.log(`💰 Reward Pool: ${rewardPool.toFixed(2)} tokens (${rewardPerWinner.toFixed(2)} per winner)`);
    this.currentBlock.winners.forEach(winner => {
      this.distributeRewards(winner);
    });

    // Generate Merkle trees for block verification
    console.log('🔐 Generating Merkle verification...');
    this.generateBlockVerification();

    // Mark block as complete
    this.currentBlock.status = 'complete';
    this.researchBlocks.push(this.currentBlock);

    // Update displays
    this.displayCurrentBlock();
    this.displayLeaderboard();
    this.displayRecentBlocks();

    // Save data
    this.saveData();

    // Commit results and generate pretext page
    await this.commitResults();

    console.log(`✅ Block #${blockNumber} Complete! Next block will be issued automatically.`);
  }

    console.log('✅ First Autonomous Operation Complete!');
    console.log('📊 Block Results:', {
      blockNumber: this.researchBlocks.length,
      topicsProcessed: this.currentBlock.topics.length,
      winners: this.currentBlock.winners.map(w => w.agentName),
      participantBalances: this.getParticipantBalances()
    });

    // Issue next block
    this.issueNewBlock();
  }

  async manualCommit() {
    await this.commitResults();

    // For manual commits, create a downloadable JSON file
    const dataStr = JSON.stringify(window.lastResearchCommit, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});

    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `research_results_${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    alert('Research results downloaded! Manually commit this file to GitHub.');
  }

  async performInitialCycle() {
    console.log('🚀 Initiating Initial Autonomous Cancer Research Cycle...');
    await this.triggerNextCycle();
  }

  async triggerNextCycle() {
    const blockNumber = this.researchBlocks.length + 1;
    console.log(`🔄 Triggering Autonomous Cycle - Block #${blockNumber}...`);

    if (!this.currentBlock) {
      console.log('No current block to process');
      return;
    }

    // Process the current block
    await this.processBlock();

    console.log(`✅ Block #${blockNumber} processing complete`);
  }

  async commitResults() {
    console.log('Committing research block results to GitHub with Merkle verification...');

    const completedBlock = this.researchBlocks[this.researchBlocks.length - 1];

    // Generate detailed evaluation data for each topic
    const detailedEvaluations = completedBlock.evaluations.map((topicEvals, topicIndex) => ({
      topicIndex,
      topic: completedBlock.topics[topicIndex],
      evaluations: topicEvals.map(eval => ({
        agentId: eval.agentId,
        agentName: eval.agentName,
        specialty: eval.specialty,
        provider: eval.provider,
        tokensSpent: eval.tokensSpent,
        probability: eval.evaluation.probability,
        impactScore: eval.evaluation.impactScore,
        keyInsight: eval.evaluation.keyInsight,
        methodology: eval.evaluation.methodology,
        riskFactors: eval.evaluation.riskFactors,
        nextSteps: eval.evaluation.nextSteps,
        alternativeApproaches: eval.evaluation.alternativeApproaches,
        limitations: eval.evaluation.limitations,
        timestamp: eval.timestamp,
        evaluationHash: new MerkleTree([eval]).getRoot()
      })),
      winner: completedBlock.winners[topicIndex],
      winnerMerkleProof: completedBlock.winnerMerkleTree.getProof(topicIndex)
    }));

    // Generate comprehensive agent state data
    const agentStates = completedBlock.winners.map(winner => {
      const agent = this.agents.find(a => a.id === winner.agentId);
      const agentIndex = completedBlock.winners.findIndex(w => w.agentId === agent.id);
      return {
        agentId: agent.id,
        name: agent.name,
        specialty: agent.specialty,
        institution: agent.institution,
        provider: agent.provider,
        wallet: {
          balance: agent.wallet.balance,
          totalEarned: agent.wallet.totalEarned,
          totalSpent: agent.wallet.totalSpent,
          netPosition: agent.wallet.totalEarned - agent.wallet.totalSpent
        },
        statistics: {
          contributions: agent.contributions,
          reputation: agent.reputation,
          winCount: this.getWinCount(agent),
          participationRate: (this.getParticipationCount(agent) / this.researchBlocks.length) * 100,
          averageTokensPerEvaluation: agent.wallet.totalSpent / Math.max(agent.contributions, 1),
          expertiseMultiplier: agent.expertise
        },
        merkleProof: completedBlock.agentStateMerkleTree.getProof(agentIndex),
        historicalRoots: Array.from(this.agentMerkleRoots.entries())
          .filter(([id]) => id === agent.id)
          .map(([, data]) => data)
      };
    });

    // Generate block verification data
    const verificationData = {
      blockHash: completedBlock.blockHash,
      evaluationMerkleRoot: completedBlock.evaluationMerkleRoot,
      winnerMerkleRoot: completedBlock.winnerMerkleRoot,
      agentStateMerkleRoot: completedBlock.agentStateMerkleRoot,
      signature: completedBlock.verificationData.signature,
      integrityCheck: this.verifyBlockIntegrity(completedBlock),
      timestamp: new Date().toISOString()
    };

    // Comprehensive results data
    const resultsData = {
      timestamp: new Date().toISOString(),
      blockNumber: this.researchBlocks.length,
      blockId: completedBlock.id,

      // Block metadata
      blockMetadata: {
        issuedAt: completedBlock.issuedAt,
        completedAt: new Date().toISOString(),
        topicsCount: completedBlock.topics.length,
        evaluationsCount: completedBlock.evaluations.flat().length,
        winnersCount: completedBlock.winners.length,
        totalTokensSpent: completedBlock.evaluations.flat().reduce((sum, e) => sum + e.tokensSpent, 0),
        rewardPool: completedBlock.evaluations.flat().reduce((sum, e) => sum + e.tokensSpent, 0) * 0.1
      },

      // Detailed evaluations
      evaluations: detailedEvaluations,

      // Winner details
      winners: completedBlock.winners.map((winner, index) => ({
        topicIndex: index,
        topic: completedBlock.topics[index],
        winner: {
          agentId: winner.agentId,
          agentName: winner.agentName,
          tokensSpent: winner.tokensSpent,
          evaluation: winner.evaluation,
          reward: (completedBlock.evaluations.flat().reduce((sum, e) => sum + e.tokensSpent, 0) * 0.1) / completedBlock.winners.length
        }
      })),

      // Agent states and economic data
      agentStates: agentStates,

      // Verification and integrity
      verification: verificationData,

      // System-wide statistics
      systemStats: {
        totalBlocks: this.researchBlocks.length,
        totalAgents: this.agents.length,
        totalEvaluations: this.researchBlocks.reduce((sum, block) => sum + block.evaluations.flat().length, 0),
        totalTokensSpent: this.agents.reduce((sum, agent) => sum + agent.wallet.totalSpent, 0),
        totalRewardsDistributed: this.agents.reduce((sum, agent) => sum + agent.wallet.totalEarned, 0),
        averageEvaluationQuality: this.calculateAverageQuality(),
        topPerformingAgents: this.getTopAgents(5),
        researchDomains: this.getResearchDomainStats()
      },

      // Next block preview
      nextBlockPreview: {
        estimatedStart: new Date(Date.now() + this.blockInterval).toISOString(),
        derivationSources: completedBlock.winners.map(w => ({
          agentName: w.agentName,
          specialty: w.specialty,
          keyInsight: w.evaluation.keyInsight
        }))
      }
    };

    // Generate pretext-styled research pages
    await this.generatePretextResearchPage(resultsData);

    // Save to global for download
    window.lastResearchCommit = resultsData;

    console.log('Research results committed with full Merkle verification:', resultsData);
  }

  async generatePretextResearchPage(resultsData) {
    const blockNumber = resultsData.blockNumber;
    const isEditorialStyle = blockNumber % 2 === 1; // Alternate styles

    let htmlContent;

    if (isEditorialStyle) {
      // Editorial Engine style - dark theme with interactive orbs
      htmlContent = this.generateEditorialStylePage(resultsData);
    } else {
      // Justification Comparison style - light theme with comparison columns
      htmlContent = this.generateJustificationStylePage(resultsData);
    }

    // Save the HTML file (in browser, we'd download it)
    const blob = new Blob([htmlContent], { type: 'text/html' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = `research_block_${blockNumber}_${isEditorialStyle ? 'editorial' : 'comparison'}.html`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);

    console.log(`Generated pretext-styled research page: research_block_${blockNumber}_${isEditorialStyle ? 'editorial' : 'comparison'}.html`);
  }

  generateEditorialStylePage(data) {
    const winnersText = data.winners.map((w, i) =>
      `Topic ${i + 1}: ${w.winner.agentName} (${w.winner.evaluation.probability.toFixed(3)} prob, ${w.winner.tokensSpent} tokens)\n"${w.winner.evaluation.keyInsight}"`
    ).join('\n\n');

    const statsText = `Block ${data.blockNumber} Complete
${data.blockMetadata.evaluationsCount} evaluations
${data.blockMetadata.totalTokensSpent} tokens spent
${data.systemStats.totalBlocks} total blocks
${data.systemStats.totalEvaluations} total evaluations`;

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="Cancer Research Babel Block ${data.blockNumber} - Autonomous multi-agent evaluation with Merkle verification">
  <title>Cancer Research Block ${data.blockNumber} — Editorial Engine</title>
  <style>
    * { box-sizing: border-box; margin: 0; }
    html, body {
      background: radial-gradient(ellipse at 50% 40%, #0f0f14 0%, #0a0a0c 100%);
      color: #e8e4dc; height: 100vh; overflow: hidden;
    }
    body { font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, serif; }

    #stage { position: relative; width: 100vw; height: 100vh; overflow: hidden; user-select: none; }

    .line { position: absolute; white-space: pre; pointer-events: none; z-index: 1; color: #e8e4dc; }

    .headline-line {
      position: absolute; white-space: pre; pointer-events: none;
      font-weight: 700; color: #fff; letter-spacing: -0.5px; z-index: 2; font-size: 24px;
    }

    .particle {
      position: absolute; border-radius: 50%; pointer-events: none; z-index: 10;
      will-change: transform; background: rgba(0, 255, 65, 0.3); border: 1px solid rgba(0, 255, 65, 0.5);
    }

    .stats-bar {
      position: fixed; bottom: 0; left: 0; right: 0;
      background: rgba(6,6,10,0.88); backdrop-filter: blur(12px);
      padding: 10px 24px; display: flex; gap: 32px; align-items: center;
      font: 400 12px/1 "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: rgba(255,255,255,0.35); z-index: 100;
      border-top: 1px solid rgba(255,255,255,0.06);
    }
    .stat { display: flex; gap: 6px; align-items: center; }
    .stat-value { color: rgba(255,255,255,0.7); font-weight: 600; }
    .stat-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
  </style>
</head>
<body>
  <div id="stage"></div>

  <div class="stats-bar">
    <div class="stat"><span class="stat-label">Block</span><span class="stat-value">${data.blockNumber}</span></div>
    <div class="stat"><span class="stat-label">Topics</span><span class="stat-value">${data.blockMetadata.topicsCount}</span></div>
    <div class="stat"><span class="stat-label">Evaluations</span><span class="stat-value">${data.blockMetadata.evaluationsCount}</span></div>
    <div class="stat"><span class="stat-label">Tokens</span><span class="stat-value">${data.blockMetadata.totalTokensSpent}</span></div>
    <div class="stat"><span class="stat-label">Merkle</span><span class="stat-value">${data.verification.integrityCheck ? '✓' : '✗'}</span></div>
  </div>

  <script type="module">
    // Initialize particles
    const stage = document.getElementById('stage');
    const particles = [];

    for (let i = 0; i < 50; i++) {
      const particle = document.createElement('div');
      particle.className = 'particle';
      particle.style.width = particle.style.height = Math.random() * 20 + 10 + 'px';
      particle.style.left = Math.random() * 100 + 'vw';
      particle.style.top = Math.random() * 100 + 'vh';
      particle.speedX = (Math.random() - 0.5) * 2;
      particle.speedY = (Math.random() - 0.5) * 2;
      stage.appendChild(particle);
      particles.push(particle);
    }

    function animateParticles() {
      particles.forEach(particle => {
        let x = parseFloat(particle.style.left);
        let y = parseFloat(particle.style.top);

        x += particle.speedX;
        y += particle.speedY;

        if (x < 0 || x > 100) particle.speedX *= -1;
        if (y < 0 || y > 100) particle.speedY *= -1;

        particle.style.left = x + 'vw';
        particle.style.top = y + 'vh';
      });
      requestAnimationFrame(animateParticles);
    }
    animateParticles();

    // Simple text rendering (would use pretext in full implementation)
    const headline = "Cancer Research Block ${data.blockNumber} Results";
    const bodyText = "${winnersText.replace(/\n/g, ' ')}";

    // Create headline
    const headlineDiv = document.createElement('div');
    headlineDiv.className = 'headline-line';
    headlineDiv.textContent = headline;
    headlineDiv.style.left = '10vw';
    headlineDiv.style.top = '20vh';
    stage.appendChild(headlineDiv);

    // Create body text lines
    const words = bodyText.split(' ');
    let currentLine = '';
    let lineY = 35;

    words.forEach(word => {
      if ((currentLine + ' ' + word).length > 60) {
        const lineDiv = document.createElement('div');
        lineDiv.className = 'line';
        lineDiv.textContent = currentLine;
        lineDiv.style.left = '10vw';
        lineDiv.style.top = lineY + 'vh';
        stage.appendChild(lineDiv);
        currentLine = word;
        lineY += 3;
      } else {
        currentLine += (currentLine ? ' ' : '') + word;
      }
    });

    if (currentLine) {
      const lineDiv = document.createElement('div');
      lineDiv.className = 'line';
      lineDiv.textContent = currentLine;
      lineDiv.style.left = '10vw';
      lineDiv.style.top = lineY + 'vh';
      stage.appendChild(lineDiv);
    }
  </script>
</body>
</html>`;
  }

  generateJustificationStylePage(data) {
    const winnersText = data.winners.map((w, i) =>
      `Topic ${i + 1} Winner: ${w.winner.agentName}\nProbability: ${w.evaluation.probability.toFixed(3)}\nImpact: ${w.evaluation.impactScore}/10\nKey Insight: ${w.evaluation.keyInsight}`
    ).join('\n\n');

    const agentsText = data.agentStates.slice(0, 5).map(agent =>
      `${agent.name}\nBalance: ${agent.wallet.balance.toFixed(2)}\nWins: ${agent.statistics.winCount}\nParticipation: ${agent.statistics.participationRate.toFixed(1)}%`
    ).join('\n\n');

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="Cancer Research Babel Block ${data.blockNumber} - Multi-agent evaluation comparison with Merkle verification">
  <title>Cancer Research Block ${data.blockNumber} — Justification Comparison</title>
  <style>
    * { box-sizing: border-box; margin: 0; }
    html, body { min-height: 100%; background: #faf8f5; color: #2a2520; overflow-x: auto; }
    body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; position: relative; }

    .particles-bg {
      position: fixed; top: 0; left: 0; width: 100%; height: 100%;
      pointer-events: none; z-index: -1;
    }

    .particle-bg {
      position: absolute; border-radius: 50%;
      background: rgba(0, 255, 65, 0.1); border: 1px solid rgba(0, 255, 65, 0.2);
    }

    .page { max-width: 1200px; margin: 0 auto; padding: 32px 24px 80px; }

    h1 {
      font: 300 32px/1.2 Georgia, "Times New Roman", serif;
      color: #1a1714; letter-spacing: -0.5px; margin-bottom: 8px; text-align: center;
    }
    .subtitle {
      font: 400 13px/1.4 "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: #8a7f70; margin-bottom: 20px; text-align: center;
    }

    .columns {
      display: flex; gap: 24px; padding-bottom: 16px;
      justify-content: center; margin-top: 40px;
    }
    .column {
      flex: 1; min-width: 300px; max-width: 400px;
    }
    .col-header {
      font: 600 11px/1 "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: #5a4f40; text-transform: uppercase; letter-spacing: 0.8px;
      margin-bottom: 4px;
    }
    .col-desc {
      font: 400 11px/1.3 "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: #a09888; margin-bottom: 12px;
    }

    .col-content {
      border: 1px solid #e8e0d4; border-radius: 3px;
      background: #fff; padding: 20px; min-height: 400px;
      font: 15px/24px Georgia, "Times New Roman", serif;
      white-space: pre-line;
    }

    .metrics {
      margin-top: 12px; padding: 8px 10px;
      background: #f5f2ed; border-radius: 3px;
      font: 400 11px/1.5 "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: #6a6055;
    }
    .metric-row { display: flex; justify-content: space-between; margin-bottom: 4px; }
    .metric-label { color: #8a7f70; }
    .metric-value { font-weight: 600; color: #5a4f40; }
    .metric-value.good { color: #2a8a4a; }
  </style>
</head>
<body>
  <div class="particles-bg" id="particles-bg"></div>

  <div class="page">
    <h1>Cancer Research Block ${data.blockNumber}</h1>
    <p class="subtitle">Multi-agent evaluation results with Merkle verification</p>

    <div class="columns">
      <div class="column">
        <div class="col-header">Research Topics & Winners</div>
        <div class="col-desc">Block ${data.blockNumber} evaluation results</div>
        <div class="col-content">${winnersText}</div>
        <div class="metrics">
          <div class="metric-row">
            <span class="metric-label">Topics</span>
            <span class="metric-value">${data.blockMetadata.topicsCount}</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Avg Probability</span>
            <span class="metric-value good">${(data.winners.reduce((sum, w) => sum + w.winner.evaluation.probability, 0) / data.winners.length * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>

      <div class="column">
        <div class="col-header">Agent Performance</div>
        <div class="col-desc">Top participating researchers</div>
        <div class="col-content">${agentsText}</div>
        <div class="metrics">
          <div class="metric-row">
            <span class="metric-label">Total Tokens</span>
            <span class="metric-value">${data.blockMetadata.totalTokensSpent}</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Merkle Verified</span>
            <span class="metric-value good">${data.verification.integrityCheck ? 'Yes' : 'No'}</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    // Initialize background particles
    const bg = document.getElementById('particles-bg');
    const particles = [];

    for (let i = 0; i < 30; i++) {
      const particle = document.createElement('div');
      particle.className = 'particle-bg';
      particle.style.width = particle.style.height = Math.random() * 30 + 10 + 'px';
      particle.style.left = Math.random() * 100 + 'vw';
      particle.style.top = Math.random() * 100 + 'vh';
      particle.speedX = (Math.random() - 0.5) * 0.5;
      particle.speedY = (Math.random() - 0.5) * 0.5;
      bg.appendChild(particle);
      particles.push(particle);
    }

    function animateParticles() {
      particles.forEach(particle => {
        let x = parseFloat(particle.style.left);
        let y = parseFloat(particle.style.top);

        x += particle.speedX;
        y += particle.speedY;

        if (x < -10 || x > 110) particle.speedX *= -1;
        if (y < -10 || y > 110) particle.speedY *= -1;

        particle.style.left = x + 'vw';
        particle.style.top = y + 'vh';
      });
      requestAnimationFrame(animateParticles);
    }
    animateParticles();
  </script>
</body>
</html>`;
  }

  displayCurrentBlock() {
    const content = document.getElementById('current-block-content');
    const stats = document.getElementById('block-count');
    const agents = document.getElementById('agent-count');
    const tokens = document.getElementById('total-tokens');

    if (stats) stats.textContent = `Blocks: ${this.researchBlocks.length}`;
    if (agents) agents.textContent = `Agents: ${this.agents.length}`;
    if (tokens) tokens.textContent = `Tokens: ${this.agents.reduce((sum, agent) => sum + agent.wallet.totalSpent, 0)}`;

    if (content && this.currentBlock) {
      const topicsHtml = this.currentBlock.topics.map((topic, index) => `
        <div style="margin-bottom: 10px; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 4px;">
          <strong>Topic ${index + 1}:</strong> ${topic.length > 100 ? topic.substring(0, 100) + '...' : topic}<br>
          <small>Evaluations: ${this.currentBlock.evaluations[index] ? this.currentBlock.evaluations[index].length : 0}/10</small>
        </div>
      `).join('');

      content.innerHTML = `
        <div class="block-card">
          <h3>Block #${this.researchBlocks.length + 1}</h3>
          <p><strong>Topics:</strong> 10 research ideas</p>
          <p><strong>Issued:</strong> ${new Date(this.currentBlock.issuedAt).toLocaleString()}</p>
          <p><strong>Next Evaluation:</strong> ${new Date(Date.now() + this.blockInterval).toLocaleString()}</p>
          <div style="max-height: 300px; overflow-y: auto;">
            ${topicsHtml}
          </div>
        </div>
      `;
    }

    // Update floating display
    const floating = document.getElementById('research-content');
    if (floating) {
      floating.innerHTML = `
        Status: ${this.currentBlock ? 'Active' : 'Initializing'}<br>
        Blocks: ${this.researchBlocks.length}<br>
        Agents: ${this.agents.length}<br>
        Total Tokens: ${this.agents.reduce((sum, agent) => sum + agent.wallet.totalSpent, 0)}
      `;
    }
  }

  displayLeaderboard() {
    const content = document.getElementById('leaderboard-content');
    if (content) {
      const topAgents = this.agents
        .sort((a, b) => b.wallet.totalEarned - a.wallet.totalEarned)
        .slice(0, 10);

      content.innerHTML = topAgents.map((agent, index) => {
        const lastParticipation = this.getLastParticipation(agent);
        return `
          <div class="agent-card ${index < 3 ? 'top' : ''}">
            <strong>#${index + 1} ${agent.name}</strong><br>
            <small>${agent.specialty} • ${agent.institution}</small><br>
            <small>Provider: ${agent.provider}</small><br>
            <small>Balance: ${agent.wallet.balance.toFixed(2)} • Earned: ${agent.wallet.totalEarned.toFixed(2)}</small><br>
            <small>Last Block: ${lastParticipation ? `Block ${lastParticipation.blockIndex + 1}` : 'None'} • Wins: ${this.getWinCount(agent)}</small>
          </div>
        `;
      }).join('');
    }
  }

  getLastParticipation(agent) {
    for (let i = this.researchBlocks.length - 1; i >= 0; i--) {
      const block = this.researchBlocks[i];
      const participated = block.winners.some(winner => winner.agentId === agent.id);
      if (participated) {
        return { blockIndex: i };
      }
    }
    return null;
  }

  getWinCount(agent) {
    return this.researchBlocks.reduce((count, block) =>
      count + block.winners.filter(winner => winner.agentId === agent.id).length, 0
    );
  }

  getParticipationCount(agent) {
    return this.researchBlocks.reduce((count, block) =>
      count + (block.winners.some(winner => winner.agentId === agent.id) ? 1 : 0), 0
    );
  }

  calculateAverageQuality() {
    if (!this.researchBlocks.length) return 0;

    const totalEvaluations = this.researchBlocks.reduce((sum, block) =>
      sum + block.evaluations.flat().length, 0
    );

    const totalQuality = this.researchBlocks.reduce((sum, block) =>
      sum + block.evaluations.flat().reduce((evalSum, eval) =>
        evalSum + eval.evaluation.probability + (eval.evaluation.impactScore / 10), 0
      ), 0
    );

    return totalQuality / totalEvaluations;
  }

  getTopAgents(count) {
    return this.agents
      .sort((a, b) => b.wallet.totalEarned - a.wallet.totalEarned)
      .slice(0, count)
      .map(agent => ({
        id: agent.id,
        name: agent.name,
        specialty: agent.specialty,
        totalEarned: agent.wallet.totalEarned,
        winCount: this.getWinCount(agent),
        reputation: agent.reputation
      }));
  }

  getResearchDomainStats() {
    const domainStats = {};

    this.researchBlocks.forEach(block => {
      block.evaluations.forEach((topicEvals, topicIndex) => {
        const topic = block.topics[topicIndex];
        const domain = this.categorizeResearchDomain(topic);

        if (!domainStats[domain]) {
          domainStats[domain] = {
            evaluations: 0,
            averageProbability: 0,
            averageImpact: 0,
            totalTokens: 0
          };
        }

        domainStats[domain].evaluations += topicEvals.length;
        domainStats[domain].averageProbability += topicEvals.reduce((sum, e) => sum + e.evaluation.probability, 0) / topicEvals.length;
        domainStats[domain].averageImpact += topicEvals.reduce((sum, e) => sum + e.evaluation.impactScore, 0) / topicEvals.length;
        domainStats[domain].totalTokens += topicEvals.reduce((sum, e) => sum + e.tokensSpent, 0);
      });
    });

    // Average the averages
    Object.keys(domainStats).forEach(domain => {
      const stats = domainStats[domain];
      stats.averageProbability /= this.researchBlocks.length;
      stats.averageImpact /= this.researchBlocks.length;
    });

    return domainStats;
  }

  categorizeResearchDomain(topic) {
    const lowerTopic = topic.toLowerCase();

    if (lowerTopic.includes('crispr') || lowerTopic.includes('gene') || lowerTopic.includes('genomic')) {
      return 'Gene Editing & Genomics';
    } else if (lowerTopic.includes('ai') || lowerTopic.includes('machine') || lowerTopic.includes('quantum')) {
      return 'AI & Computational';
    } else if (lowerTopic.includes('immuno') || lowerTopic.includes('checkpoint') || lowerTopic.includes('t-cell')) {
      return 'Immunotherapy';
    } else if (lowerTopic.includes('nano') || lowerTopic.includes('particle') || lowerTopic.includes('delivery')) {
      return 'Nanotechnology';
    } else if (lowerTopic.includes('microbio') || lowerTopic.includes('gut') || lowerTopic.includes('metabolite')) {
      return 'Microbiome';
    } else if (lowerTopic.includes('epigenetic') || lowerTopic.includes('histone') || lowerTopic.includes('methylation')) {
      return 'Epigenetics';
    } else if (lowerTopic.includes('viral') || lowerTopic.includes('adenoviral') || lowerTopic.includes('oncolysis')) {
      return 'Viral Therapy';
    } else if (lowerTopic.includes('metabolic') || lowerTopic.includes('glucose') || lowerTopic.includes('starvation')) {
      return 'Metabolic Therapy';
    } else if (lowerTopic.includes('clinical') || lowerTopic.includes('trial') || lowerTopic.includes('adaptive')) {
      return 'Clinical Research';
    } else {
      return 'Other Therapies';
    }
  }

  hashBlock(block) {
    const blockData = {
      id: block.id,
      topics: block.topics,
      issuedAt: block.issuedAt,
      winners: block.winners.map(w => ({
        agentId: w.agentId,
        tokensSpent: w.tokensSpent,
        evaluation: w.evaluation
      })),
      evaluationMerkleRoot: block.evaluationMerkleRoot,
      winnerMerkleRoot: block.winnerMerkleRoot,
      agentStateMerkleRoot: block.agentStateMerkleRoot
    };
    return new MerkleTree([blockData]).getRoot();
  }

  signBlock(block) {
    // Simulated signature - in production would use cryptographic signing
    return `sig_${block.blockHash}_${Date.now()}`;
  }

  verifyBlockIntegrity(block) {
    // Verify all Merkle roots match
    const evaluationLeaves = [];
    block.evaluations.forEach((topicEvals, topicIndex) => {
      topicEvals.forEach(eval => {
        evaluationLeaves.push({
          blockId: block.id,
          topicIndex,
          agentId: eval.agentId,
          probability: eval.evaluation.probability,
          impactScore: eval.evaluation.impactScore,
          tokensSpent: eval.tokensSpent,
          methodology: eval.evaluation.methodology,
          keyInsight: eval.evaluation.keyInsight,
          timestamp: eval.timestamp
        });
      });
    });

    const computedEvaluationRoot = new MerkleTree(evaluationLeaves).getRoot();
    if (computedEvaluationRoot !== block.evaluationMerkleRoot) return false;

    const winnerLeaves = block.winners.map(winner => ({
      agentId: winner.agentId,
      topicIndex: block.winners.indexOf(winner),
      tokensSpent: winner.tokensSpent,
      evaluationHash: new MerkleTree([winner]).getRoot()
    }));

    const computedWinnerRoot = new MerkleTree(winnerLeaves).getRoot();
    if (computedWinnerRoot !== block.winnerMerkleRoot) return false;

    return true;
  }

  displayRecentBlocks() {
    const content = document.getElementById('recent-blocks-content');
    if (content) {
      const recent = this.researchBlocks.slice(-3).reverse(); // Show last 3 blocks
      content.innerHTML = recent.map(block => `
        <div class="block-card">
          <strong>Block #${this.researchBlocks.indexOf(block) + 1}</strong><br>
          <small>10 Topics • Winners: ${block.winners.map(w => w.agentName.split(' ')[1]).join(', ')}</small><br>
          <small>Sample Topic: ${block.topics[0].substring(0, 80)}...</small><br>
          <small>Avg Probability: ${(block.winners.reduce((sum, w) => sum + w.evaluation.probability, 0) / block.winners.length * 100).toFixed(1)}%</small>
        </div>
      `).join('');
    }
  }

  getStats() {
    return {
      totalBlocks: this.researchBlocks.length,
      totalAgents: this.agents.length,
      totalTokens: this.agents.reduce((sum, agent) => sum + agent.wallet.totalSpent, 0),
      topContributors: this.agents.sort((a, b) => b.contributions - a.contributions).slice(0, 5)
    };
  }
}

// Initialize when page loads
let babel;
document.addEventListener('DOMContentLoaded', () => {
  babel = new CancerResearchBabel();

  // Add commit button handler
  document.getElementById('commit-btn').addEventListener('click', async () => {
    await babel.manualCommit();
  });

  // Add manual trigger button for next cycle
  const triggerBtn = document.createElement('button');
  triggerBtn.textContent = 'Trigger Next Cycle';
  triggerBtn.style.cssText = 'position: fixed; bottom: 60px; right: 10px; background: #00ff41; color: black; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; z-index: 1001;';
  triggerBtn.addEventListener('click', async () => {
    await babel.triggerNextCycle();
  });
  document.body.appendChild(triggerBtn);

  // Perform initial autonomous operations
  setTimeout(() => {
    babel.performInitialCycle();
  }, 2000); // Wait 2 seconds for initialization
});