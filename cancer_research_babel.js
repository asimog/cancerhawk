class CancerResearchBabel {
  constructor() {
    this.seedCode = "e676098e48313c65989af66900ba43461e738168c3bab7ffbb12e0c96319ed9b6a0e6f5fe1f1737291148c25db2a565b35cf6fe9b00d997c8c9879e735fa81d8SN_3a7c39f14e0c478bc1b8b33ee4e7b4d18e2c8659e7a37f2f5b1b464c8b6f5b19RA_93afb3fdc190a2c9450f42a5e55474db9f69f3630cbd33b783af33c23093742f";
    this.agents = [];
    this.researchBlocks = [];
    this.currentBlock = null;
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

  processBlock() {
    if (!this.currentBlock || this.currentBlock.status !== 'active') return;

    // Select 10 random agents for this block
    const selectedAgents = this.selectRandomAgents(this.agentsPerBlock);

    // Each agent evaluates all 10 topics in the block
    this.currentBlock.evaluations = this.currentBlock.topics.map(topic =>
      selectedAgents.map(agent => ({
        ...this.simulateEvaluation(agent, topic),
        participated: true
      }))
    );

    // Determine winner for each topic (highest token spend among evaluators)
    this.currentBlock.winners = this.currentBlock.evaluations.map(topicEvaluations => {
      const winner = topicEvaluations.reduce((prev, current) =>
        (prev.tokensSpent > current.tokensSpent) ? prev : current
      );
      return winner;
    });

    // Distribute rewards for each topic
    this.currentBlock.winners.forEach(winner => {
      this.distributeRewards(winner);
    });

    // Mark block as complete
    this.currentBlock.status = 'complete';
    this.researchBlocks.push(this.currentBlock);

    // Update displays
    this.displayCurrentBlock();
    this.displayLeaderboard();
    this.displayRecentBlocks();

    // Save and commit
    this.saveData();
    this.commitResults();
  }

  selectRandomAgents(count) {
    const shuffled = [...this.agents].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
  }

  simulateEvaluation(agent, topic) {
    // Simulate token spending based on expertise and provider
    const baseTokens = 100;
    const expertiseMultiplier = agent.expertise;
    const providerMultiplier = this.getProviderMultiplier(agent.provider);
    const tokensSpent = Math.floor(baseTokens * expertiseMultiplier * providerMultiplier * (0.8 + Math.random() * 0.4));

    // Simulate evaluation
    const probability = Math.min(1, Math.max(0, (agent.expertise * 0.7 + Math.random() * 0.3)));
    const impact = Math.floor(probability * 10 * (0.8 + Math.random() * 0.4)) / 10;

    return {
      agentId: agent.id,
      agentName: agent.name,
      specialty: agent.specialty,
      provider: agent.provider,
      tokensSpent,
      evaluation: {
        probability,
        impactScore: impact,
        keyInsight: this.generateInsight(topic, probability),
        methodology: this.generateMethodology(agent.specialty),
        riskFactors: this.generateRiskFactors(),
        nextSteps: this.generateNextSteps(),
        alternativeApproaches: this.generateAlternatives(),
        followUpQuestion: this.generateFollowUp(topic),
        limitations: this.generateLimitations()
      },
      timestamp: new Date().toISOString()
    };
  }

  getProviderMultiplier(provider) {
    const multipliers = {
      'openai/gpt-4o': 1.2,
      'anthropic/claude-3.5-sonnet': 1.1,
      'google/gemini-2.0': 1.0,
      'meta/llama-3.1-70b': 0.9,
      'mistral/mistral-7b': 0.8
    };
    return multipliers[provider] || 1.0;
  }

  generateInsight(topic, probability) {
    const insights = [
      `The approach shows ${Math.round(probability * 100)}% potential for clinical translation`,
      `Key breakthrough in ${topic.split(' ')[0]} mechanism understanding`,
      `Novel combination therapy could overcome resistance patterns`,
      `Biomarker discovery enables patient stratification`,
      `Therapeutic window identified for safe administration`
    ];
    return insights[Math.floor(Math.random() * insights.length)];
  }

  generateMethodology(specialty) {
    const methodologies = {
      'Molecular Oncology': 'CRISPR screening and proteomics',
      'Immunotherapy': 'T-cell receptor sequencing and flow cytometry',
      'Gene Therapy': 'Viral vector engineering and delivery optimization',
      'Clinical Trials': 'Adaptive trial design and real-world evidence',
      'Translational Research': 'Patient-derived organoids and xenografts',
      'Cancer Genomics': 'Whole genome sequencing and bioinformatics',
      'Drug Development': 'High-throughput screening and medicinal chemistry',
      'default': 'Multi-omics integration and computational modeling'
    };
    return methodologies[specialty] || methodologies.default;
  }

  generateRiskFactors() {
    return [
      'Off-target effects in normal tissues',
      'Immune rejection of therapeutic constructs',
      'Tumor heterogeneity and clonal evolution',
      'Manufacturing scalability challenges',
      'Regulatory approval timelines'
    ].sort(() => Math.random() - 0.5).slice(0, 3);
  }

  generateNextSteps() {
    return [
      'Preclinical toxicity studies in multiple species',
      'Phase 1 dose-escalation clinical trial',
      'Biomarker validation in patient cohorts',
      'Manufacturing process optimization',
      'Intellectual property protection strategy'
    ].sort(() => Math.random() - 0.5).slice(0, 3);
  }

  generateAlternatives() {
    return [
      'Small molecule inhibitors',
      'Monoclonal antibody conjugates',
      'Cell-based therapies',
      'RNA interference approaches',
      'Epigenetic modulators'
    ].sort(() => Math.random() - 0.5).slice(0, 2);
  }

  generateFollowUp(topic) {
    return `How can ${topic.split(' ')[0]} approaches be combined with immunotherapy?`;
  }

  generateLimitations() {
    return [
      'Limited long-term safety data',
      'High development costs',
      'Complex regulatory pathway',
      'Patient accessibility issues',
      'Technical implementation challenges'
    ].sort(() => Math.random() - 0.5).slice(0, 2);
  }

  distributeRewards(winner) {
    const totalTokens = this.currentBlock.evaluations.reduce((sum, eval) => sum + eval.tokensSpent, 0);
    const rewardPool = totalTokens * 0.1; // 10% of total tokens as reward

    // Winner gets 50% of reward pool
    winner.agent.wallet.balance += rewardPool * 0.5;
    winner.agent.wallet.totalEarned += rewardPool * 0.5;

    // Other agents get proportional rewards based on their contribution
    this.currentBlock.evaluations.forEach(eval => {
      if (eval.agentId !== winner.agentId) {
        const proportion = eval.tokensSpent / totalTokens;
        const reward = rewardPool * 0.5 * proportion;
        eval.agent.wallet.balance += reward;
        eval.agent.wallet.totalEarned += reward;
      }
    });

    // Update contributions
    this.agents.forEach(agent => {
      const eval = this.currentBlock.evaluations.find(e => e.agentId === agent.id);
      if (eval) {
        agent.contributions++;
        agent.wallet.totalSpent += eval.tokensSpent;
      }
    });
  }

  async commitResults() {
    console.log('Committing research block results to GitHub...');

    // Save current state to file
    const resultsData = {
      timestamp: new Date().toISOString(),
      blockNumber: this.researchBlocks.length,
      currentBlock: this.currentBlock,
      completedBlock: this.researchBlocks[this.researchBlocks.length - 1],
      participantBalances: this.getParticipantBalances(),
      agentStats: this.agents.map(agent => ({
        id: agent.id,
        name: agent.name,
        specialty: agent.specialty,
        institution: agent.institution,
        provider: agent.provider,
        balance: agent.wallet.balance,
        totalSpent: agent.wallet.totalSpent,
        totalEarned: agent.wallet.totalEarned,
        contributions: agent.contributions,
        reputation: agent.reputation,
        winCount: this.getWinCount(agent),
        lastParticipation: this.getLastParticipation(agent)
      })),
      systemStats: this.getStats()
    };

    // Save to local file (in browser, this would be download or local storage)
    window.lastResearchCommit = resultsData;

    console.log('Research results committed:', resultsData);
  }

  getParticipantBalances() {
    if (!this.researchBlocks.length) return {};

    const lastBlock = this.researchBlocks[this.researchBlocks.length - 1];
    const participants = new Set();

    // Collect all participants from the last block
    lastBlock.winners.forEach(winner => participants.add(winner.agentId));
    lastBlock.evaluations.forEach(topicEvals =>
      topicEvals.forEach(eval => participants.add(eval.agentId))
    );

    const balances = {};
    participants.forEach(agentId => {
      const agent = this.agents.find(a => a.id === agentId);
      if (agent) {
        balances[agent.name] = {
          balance: agent.wallet.balance,
          totalEarned: agent.wallet.totalEarned,
          totalSpent: agent.wallet.totalSpent,
          specialty: agent.specialty
        };
      }
    });

    return balances;
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
});