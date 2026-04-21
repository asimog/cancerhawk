class CancerResearchBabel {
  constructor() {
    this.agents = [];
    this.researchBlocks = [];
    this.currentBlock = null;
    this.blockInterval = 1 * 60 * 1000; // 1 minute for testing, change to 10 * 60 * 1000 for production
    this.minAgents = 42;
    this.providers = ['openai/gpt-4o', 'anthropic/claude-3.5-sonnet', 'google/gemini-2.0', 'meta/llama-3.1-70b', 'mistral/mistral-7b'];
    this.loadData();
    this.initializeAgents();
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
    let topic;
    if (this.researchBlocks.length === 0) {
      // Initial topics
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
      topic = initialTopics[Math.floor(Math.random() * initialTopics.length)];
    } else {
      // Derive from previous winner
      const winner = this.researchBlocks[this.researchBlocks.length - 1].winner;
      topic = this.generateDerivedTopic(winner.evaluation);
    }

    this.currentBlock = {
      id: `block_${Date.now()}`,
      topic,
      issuedAt: new Date().toISOString(),
      evaluations: [],
      winner: null,
      status: 'active'
    };

    this.saveData();
    this.displayCurrentBlock();
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

    // All agents evaluate the current block
    this.agents.forEach(agent => {
      const evaluation = this.simulateEvaluation(agent, this.currentBlock.topic);
      this.currentBlock.evaluations.push(evaluation);
    });

    // Determine winner (highest token spend)
    const winner = this.currentBlock.evaluations.reduce((prev, current) =>
      (prev.tokensSpent > current.tokensSpent) ? prev : current
    );

    // Distribute rewards
    this.distributeRewards(winner);

    // Mark block as complete
    this.currentBlock.winner = winner;
    this.currentBlock.status = 'complete';
    this.researchBlocks.push(this.currentBlock);

    // Issue new block
    this.issueNewBlock();

    // Save and commit
    this.saveData();
    this.commitResults();
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
      currentBlock: this.currentBlock,
      recentBlocks: this.researchBlocks.slice(-5), // Last 5 blocks
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
        reputation: agent.reputation
      })),
      systemStats: this.getStats()
    };

    // Save to local file (in browser, this would be download or local storage)
    // For now, save to a global variable that can be accessed
    window.lastResearchCommit = resultsData;

    // In a real implementation with Node.js backend, this would:
    // 1. Write results to research_log.json
    // 2. Run git add research_log.json
    // 3. Run git commit -m "Research Block #X completed - Winner: Y"
    // 4. Run git push

    console.log('Research results committed:', resultsData);
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
    const content = document.getElementById('research-content');
    if (content && this.currentBlock) {
      content.innerHTML = `
        <h3>Block #${this.researchBlocks.length + 1}</h3>
        <p><strong>Topic:</strong> ${this.currentBlock.topic}</p>
        <p><strong>Issued:</strong> ${new Date(this.currentBlock.issuedAt).toLocaleString()}</p>
        <p><strong>Agents:</strong> ${this.agents.length}</p>
        <p><strong>Next:</strong> ${new Date(Date.now() + this.blockInterval).toLocaleString()}</p>
        <p><strong>Blocks Completed:</strong> ${this.researchBlocks.length}</p>
      `;
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