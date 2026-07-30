/*
 * MIT-licensed
 * Copyright (c) 2024 Weco AI Ltd
 * See THIRD_PARTY_LICENSES.md for the full licence text.
 * https://github.com/WecoAI/aideml/blob/main/LICENSE
*/

// ======================
//  p5.js + Node/Edge code
// ======================

const bgCol = "#F2F0E7";
const accentCol = "#fd4578";

hljs.initHighlightingOnLoad();

const updateTargetDims = () => {
  // width is max-width of `.contentContainer` - its padding
  return [windowWidth * (1 / 2), windowHeight];
};

const setPerformance = (performance) => {
  const perfElm = document.getElementById("performance");
  if (perfElm) {
    perfElm.innerHTML = `<strong>Performance:</strong> ${performance.toFixed(2)}`;
  }
};

const setNodeDetails = (nodeIndex) => {
  if (!treeStructData) return;

  const code = treeStructData.code[nodeIndex];
  const plan = treeStructData.plan[nodeIndex];
  const performance = treeStructData.fitnesses[nodeIndex];
  const termOut = treeStructData.term_out[nodeIndex];
  const analysis = treeStructData.analysis[nodeIndex];
  const prompt = treeStructData.prompts[nodeIndex];
  const metricInfos = treeStructData.metric_infos[nodeIndex];

  // Plan
  const planElm = document.getElementById("plan");
  if (planElm) {
    planElm.innerHTML = hljs.highlight(plan, { language: "markdown" }).value;
  }

  // Prompt
  const promptElm = document.getElementById("prompt-display");
  if (promptElm) {
    promptElm.innerHTML = hljs.highlight(prompt || "No prompt available.", { language: "plaintext" }).value;
  }

  // Code
  const codeElm = document.getElementById("code");
  if (codeElm) {
    codeElm.innerHTML = hljs.highlight(code, { language: "python" }).value;
  }

  // Term Out
  const termOutElm = document.getElementById("termOut");
  if (termOutElm) {
    termOutElm.innerHTML = hljs.highlight(termOut, {
      language: "plaintext",
    }).value;
  }

  // Analysis
  const analysisElm = document.getElementById("analysis");
  if (analysisElm) {
    analysisElm.innerHTML = hljs.highlight(analysis, {
      language: "plaintext",
    }).value;
  }

  // Performance
  setPerformance(performance);

  // Auxiliary Metrics
  const auxElm = document.getElementById("auxiliarymetrics");
  if (auxElm) {
    auxElm.innerHTML = `<strong>Auxiliary Metrics:</strong> ${metricInfos.fixed(2)}`;
  }
};

// Function to copy text to clipboard
function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(function() {
    console.log('Async: Copying to clipboard was successful!');
    // Optionally provide feedback to the user, e.g., change button text
    const btn = document.getElementById('copy-data-btn');
    if (btn) {
      const originalText = btn.innerText;
      btn.innerText = 'Copied!';
      setTimeout(() => { btn.innerText = originalText; }, 1500); // Reset after 1.5s
    }
  }, function(err) {
    console.error('Async: Could not copy text: ', err);
  });
}

windowResized = () => {
  resizeCanvas(...updateTargetDims());
  awaitingPostResizeOps = true;
};

const animEase = (t) => 1 - (1 - Math.min(t, 1.0)) ** 5;

// ---- global constants ----
const globalAnimSpeed = 5.1;
const baseScaleFactor = 0.57;

// ---- global vars ----
let globalTime = 0;
let manualSelection = false;

let currentElemInd = 0;

let treeStructData = null/*<-replacehere*/; // keep as is

function getPercentiles(scores) {
  // Filter out valid scores
  const validScores = scores.filter(s => !Number.isNaN(s));
  const n = validScores.length;

  // Edge case: if there's at most one valid score, anything valid is top (1) and NaN is 0
  if (n <= 1) {
    return scores.map(s => Number.isNaN(s) ? 0 : 1);
  }

  // Sort valid scores ascending
  const sorted = [...validScores].sort((a, b) => a - b);

  // Build a Map from score -> percentile
  const percentileMap = new Map();
  let i = 0;
  while (i < n) {
    const value = sorted[i];
    let start = i;
    let end = i;

    // Find range of duplicates
    while (end + 1 < n && sorted[end + 1] === value) {
      end++;
    }

    // Average rank of duplicates
    const avgRank = (start + end) / 2;
    // Convert to [0..1] range. (n-1) ensures the max value gets percentile = 1.
    const percentile = avgRank / (n - 1);

    percentileMap.set(value, percentile);
    i = end + 1;
  }

  // Build the result array
  return scores.map(s => {
    if (Number.isNaN(s)) {
      return 0; // Because you said: "and set 0 for NaNs"
    }
    return percentileMap.get(s);
  });
}

const treeStats = (() => {

  // Abort gracefully until the JSON blob is injected
  const safe = fn => (treeStructData && treeStructData.edges) ? fn() : null;

  /* ---------- collectors ---------- */

  function collectScores () {
    return treeStructData.node_data_list.map(function (nd) {
      return (nd && nd.metric_info) ? Number(nd.metric_info.score) : NaN;
    });
  }

  function collectBugs () {
    return treeStructData.node_data_list.map(function (nd) {
      return nd ? Boolean(nd.is_buggy) : false;
    });
  }

  /* ---------- edge iterator ---------- */

  function forEachEdge (cb){
    treeStructData.edges.forEach(function (pair){
      cb(pair[0], pair[1]);          // pair = [parentIdx, childIdx]
    });
  }

  /* ---------- public API ---------- */
  return {

    childHigherThanParent : function (){
      return safe(function (){
        const scores = collectScores();
        var hit = 0, total = 0;

        forEachEdge(function (p, c){
          var ps = scores[p], cs = scores[c];
          if (!Number.isNaN(ps) && !Number.isNaN(cs)){
            ++total;
            if (cs > ps) ++hit;
          }
        });
        return { hit: hit, total: total, pct: total ? hit/total : 0 };
      });
    },

    scoreChangeAggregates : function (){
      return safe(function (){
        const scores = collectScores();
        var diffs = [];

        forEachEdge(function (p, c){
          var dp = scores[p], dc = scores[c];
          if (!Number.isNaN(dp) && !Number.isNaN(dc))
            diffs.push(dc - dp);
        });

        var n = diffs.length;
        if (!n) return { n:0, mean:0, stdev:0, median:0,
                         maxImprovement:0, maxDrop:0 };

        var sum = diffs.reduce(function (a,b){ return a+b; }, 0);
        var mean = sum / n;
        var variance = diffs.reduce(function (a,b){
                          return a + Math.pow(b-mean,2);
                        }, 0) / n;
        var stdev = Math.sqrt(variance);

        diffs.sort(function (a,b){ return a-b; });
        var median = diffs[(n-1)>>1];
        return {
          n: n,
          mean: mean,
          stdev: stdev,
          median: median,
          maxImprovement: diffs[diffs.length-1],
          maxDrop: diffs[0]
        };
      });
    },

    bugStats : function (){
      return safe(function (){
        const bugs = collectBugs();
        var bugCount = bugs.filter(Boolean).length;
        var total = bugs.length;
        var rate  = total ? bugCount/total : 0;

        var parentBugEdges = 0, parentToChildBug = 0;
        var childrenByParent = {};

        forEachEdge(function (p, c){
          if (!childrenByParent[p]) childrenByParent[p] = [];
          childrenByParent[p].push(c);

          if (bugs[p]){
            ++parentBugEdges;
            if (bugs[c]) ++parentToChildBug;
          }
        });

        var propagation = parentBugEdges ? parentToChildBug/parentBugEdges : 0;
        var bugToCleanEdges = parentBugEdges - parentToChildBug;
        var fixRate = parentBugEdges ? bugToCleanEdges / parentBugEdges : 0;

        var siblingClusters = 0;
        Object.values(childrenByParent).forEach(function (arr){
          var buggyKids = arr.filter(function (i){ return bugs[i]; }).length;
          if (buggyKids > 1) ++siblingClusters;
        });

        return {
          bugCount: bugCount, total: total, rate: rate,
          parentBugEdges: parentBugEdges,
          parentToChildBug: parentToChildBug,
          propagation: propagation,
          siblingClusters: siblingClusters,
          bugToCleanEdges: bugToCleanEdges,
          fixRate : fixRate
        };
      });
    },

    debugDepthStats : function () {
      return safe(function () {

        /* ---------- fast look‑ups ---------- */
        const buggy     = collectBugs();                 // [bool]
        const parentOf  = {};                            // child → parent
        const children  = {};                            // parent → [children]

        treeStructData.edges.forEach(([p, c]) => {
          parentOf[c] = p;
          (children[p] = children[p] || []).push(c);
        });

        /* ---------- DFS from every chain root ---------- */
        const chainLengths = [];

        function dfs(node, depth){
          const kids = children[node] || [];
          let extended = false;

          kids.forEach(k => {
            if (buggy[k]){                // keep walking the buggy chain
              extended = true;
              dfs(k, depth + 1);
            }
          });

          if (!extended){                 // leaf of the buggy chain
            chainLengths.push(depth);
          }
        }

        buggy.forEach((isBug, idx) => {
          const par = parentOf[idx];
          if (isBug && (!buggy[par])){     // parent clean –> start of a chain
            dfs(idx, 1);
          }
        });

        const total = chainLengths.length;
        const sum   = chainLengths.reduce((a,b)=>a+b, 0);
        const avg   = total ? sum / total : 0;
        const depth10 = chainLengths.filter(l => l === 11).length;

        return { total, avg, depth10 };
      });
    }
  };
})();

// Example: if your code injects data into treeStructData, do it before setup() runs
// Add .relSize property
treeStructData.relSize = getPercentiles(treeStructData.fitnesses);

let lastClick = 0;
let firstFrameTime = undefined;

let nodes = [];
let edges = [];

let lastScrollPos = 0;

let offsetX = 0; // how far we've panned horizontally
let offsetY = 0; // how far we've panned vertically
let zoomFactor = 1.0; // current zoom level
let draggingNode = null; // the node we're dragging (if any)
let nodeDragOffsetX = 0;
let nodeDragOffsetY = 0;
let isPanning = false; // true when dragging empty space

setup = () => {
  // p5 setup
  canvas = createCanvas(...updateTargetDims());
};

function screenToWorld(px, py) {
  // Reverse the same transformations we do in draw():
  // 1) Undo translate(width/2 + offsetX, height/2 + offsetY)
  // 2) Undo scale(zoomFactor * baseScaleFactor)
  let tx = px - (width / 2 + offsetX);
  let ty = py - (height / 2 + offsetY);
  let s = zoomFactor * baseScaleFactor;

  let worldX = tx / s;
  let worldY = ty / s;

  return [worldX, worldY];
}

// We'll also want a quick helper for world->screen:
function worldToScreen(wx, wy) {
  let s = zoomFactor * baseScaleFactor;
  let sx = wx * s + (width / 2 + offsetX);
  let sy = wy * s + (height / 2 + offsetY);
  return [sx, sy];
}

class Node {
  x;
  y;
  size;
  xT;
  yT;
  xB;
  yB;
  treeInd;
  relSize;
  animationStart = Number.MAX_VALUE;
  animationProgress = 0;
  isStatic = false;
  hasChildren = false;
  isRootNode = true;
  isStarred = false;
  selected = false;
  renderSize = 10;
  edges = [];
  isBuggy = false;
  displayMetric = 0;
  nodeData = null;

  constructor(x, y, relSize, treeInd) {
    // You can adjust minSize/maxSize to allow visual size differences:
    const minSize = 18;
    const maxSize = 18;

    this.relSize = relSize;
    this.treeInd = treeInd;
    this.size = minSize + (maxSize - minSize) * relSize;

    // We'll keep x,y in WORLD coords:
    this.x = x;
    this.y = y;

    // We'll update top/bottom anchors each frame
    this.xT = x;
    this.yT = y;
    this.xB = x;
    this.yB = y;

    this.nodeData = treeStructData.node_data_list[treeInd];
    this.isBuggy = treeStructData.node_data_list[treeInd].is_buggy;
    try {
      this.displayMetric = treeStructData.node_data_list[treeInd].metric_info.score.toFixed(2);
    } catch (e) {
      this.displayMetric = '? ?';
    }
    nodes.push(this);
  }

  startAnimation = (offset = 0) => {
    if (this.animationStart === Number.MAX_VALUE) {
      this.animationStart = globalTime + offset;
    }
  };

  child = (node) => {
    let edge = new Edge(this, node);
    this.edges.push(edge);
    edges.push(edge);
    this.hasChildren = true;
    node.isRootNode = false;
    return node;
  };

  render = () => {
    if (globalTime - this.animationStart < 0) return;

    // "Base" size for the node, used for the little bounce effect
    this.renderSize = this.size;

    if (!this.isStatic) {
      this.animationProgress = animEase((globalTime - this.animationStart) / 1000);
      if (this.animationProgress >= 1) {
        this.isStatic = true;
      } else {
        // A slight bounce-in effect:
        this.renderSize = this.size * (
          0.8 +
          0.2 * (-3.33 * this.animationProgress ** 2 + 4.33 * this.animationProgress)
        );
      }
    }

    push();
    // We are already inside the main translate(...) and scale(...) from draw(),
    // so the node's (x,y) is in "world" coords. Let's move to that spot:
    translate(this.x, this.y);

    // Undo the current zoom so the node is always a fixed on-screen size:
    scale(1 / (zoomFactor * baseScaleFactor));

    // -------------------------------------------------------------------------
    // 1) Draw a subtle drop shadow behind the node
    // -------------------------------------------------------------------------
    noStroke();
    fill(0, 50); // semi-transparent black for shadow
    // Slight offset to make the shadow more visible
    ellipse(3, 4, this.renderSize * 2.2, this.renderSize * 2.2);

    // -------------------------------------------------------------------------
    // 2) Outline + fill color
    // -------------------------------------------------------------------------
    // We'll do a color gradient based on relSize:
    const c1 = color(120, 220, 120); // bright pastel green
    const c2 = color(20, 80, 20);    // darker green
    const nodeFill = lerpColor(c1, c2, this.relSize);

    strokeWeight(2);
    stroke('#333');  // dark gray outline

    if (this.selected) {
      fill(accentCol);
    } else if (this.isBuggy) {
      fill('#992200');
    } else {
      fill(nodeFill);
    }

    // Draw a circle for the node
    ellipse(0, 0, this.renderSize * 2, this.renderSize * 2);

    // -------------------------------------------------------------------------
    // 3) Label in the center (metric or "buggy" symbol)
    // -------------------------------------------------------------------------
    noStroke();
    fill(255);
    textAlign(CENTER, CENTER);
    textFont('Helvetica Neue');
    textSize(this.renderSize * 0.4);

    if (this.isBuggy) {
      text("! !", 0, 0);
    } else {
      let medal_emoji = "";
      try {
        let metric_info = this.nodeData.metric_info;
        if (metric_info.gold_medal) {
          medal_emoji = "🥇";
        } else if (metric_info.silver_medal) {
          medal_emoji = "🥈";
        } else if (metric_info.bronze_medal) {
          medal_emoji = "🥉";
        } else if (metric_info.above_median) {
          medal_emoji = "👌";
        }
      } catch (e) {
        // no-op
      }
      // Show numeric score + optional emoji
      text(`${this.displayMetric}\n${medal_emoji}`, 0, 0);
    }

    // -------------------------------------------------------------------------
    // 4) If starred, show a star (example from original code)
    // -------------------------------------------------------------------------
    const dotAnimThreshold = 0.85;
    if (this.isStarred && this.animationProgress >= dotAnimThreshold) {
      let dotAnimProgress =
        (this.animationProgress - dotAnimThreshold) / (1 - dotAnimThreshold);
      textSize(
        ((-3.33 * dotAnimProgress ** 2 + 4.33 * dotAnimProgress) * this.renderSize) / 2
      );
      if (this.selected) {
        fill(0);
        stroke(0);
      } else {
        fill(accentCol);
        stroke(accentCol);
      }
      strokeWeight((-(dotAnimProgress ** 2) + dotAnimProgress) * 2);
      text("*", this.renderSize * 0.5, -this.renderSize * 0.3);
      noStroke();
    }

    // -------------------------------------------------------------------------
    // 5) "Cover" for animation effect (optional from original code)
    // -------------------------------------------------------------------------
    if (!this.isStatic) {
      fill(bgCol);
      const progressAnimBaseSize = this.renderSize + 5;
      rect(
        -progressAnimBaseSize / 2,
        -progressAnimBaseSize / 2 +
          progressAnimBaseSize * this.animationProgress,
        progressAnimBaseSize,
        progressAnimBaseSize * (1 - this.animationProgress)
      );
    }

    pop(); // end push

    // *** Update top/bottom anchor points for edges ***
    const halfScaled = (this.renderSize) / (zoomFactor * baseScaleFactor);
    this.xT = this.x;
    this.yT = this.y - halfScaled;
    this.xB = this.x;
    this.yB = this.y + halfScaled;

    // Possibly start child edges if we're near the end of the anim
    if (this.animationProgress >= 0.9) {
      this.edges
        .sort((a, b) => a.color() - b.color())
        .forEach((e, i) => {
          e.startAnimation((i / this.edges.length) ** 2 * 1000);
        });
    }
  };
}

class Edge {
  nodeT;
  nodeB;
  animationStart = Number.MAX_VALUE;
  animationProgress = 0;
  isStatic = false;
  weight = 0;

  constructor(nodeT, nodeB) {
    this.nodeT = nodeT;
    this.nodeB = nodeB;
    this.weight = 1.5 + nodeB.relSize * 1;
  }

  color = () => this.nodeB.relSize;

  startAnimation = (offset = 0) => {
    if (this.animationStart === Number.MAX_VALUE) {
      this.animationStart = globalTime + offset;
    }
  };

  render = () => {
    // If the globalTime is earlier than the start, just skip
    if (globalTime < this.animationStart) return;

    let t = (globalTime - this.animationStart) / 1000;
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    this.animationProgress = animEase(t);
    if (this.animationProgress >= 1) {
      this.isStatic = true;
    }

    // The parent's side is always up-to-date:
    const startX = this.nodeT.xB;
    const startY = this.nodeT.yB;

    // The child's final position (the node's top anchor):
    const finalX = this.nodeB.xT;
    const finalY = this.nodeB.yT;

    // We'll compute current child coords depending on animation
    let curX, curY;
    if (!this.isStatic) {
      // While still animating, pick a point along a bezier
      curX = bezierPoint(startX, startX, finalX, finalX, this.animationProgress);
      curY = bezierPoint(
        startY,
        (startY + finalY) / 2,
        (startY + finalY) / 2,
        finalY,
        this.animationProgress
      );
    } else {
      // Once static, just snap to the child's actual top anchor
      curX = finalX;
      curY = finalY;
    }

    // If the edge has nearly completed, start animating the child
    if (this.animationProgress >= 0.97) {
      this.nodeB.startAnimation();
    }

    // Draw the edge
    push();
    let thickness = this.weight / (zoomFactor * baseScaleFactor);
    strokeWeight(thickness);
    // Subtle line color
    stroke(
      lerpColor(color('rgba(0,0,0,0.2)'), color('rgba(0,0,0,0.2)'), this.nodeB.relSize * 1 + 0.7)
    );
    noFill();
    bezier(
      startX, startY,
      startX, (startY + finalY) / 2,
      curX,   (startY + finalY) / 2,
      curX,   curY
    );
    pop();
  };
}

// p5 draw:
draw = () => {
  cursor(ARROW);
  frameRate(120);

  if (!firstFrameTime && frameCount <= 1) {
    firstFrameTime = millis();
  }

  // ---- update global animation state ----
  const initialSpeedScalingEaseIO =
    (cos(min((millis() - firstFrameTime) / 8000, 1.0) * PI) + 1) / 2;
  const initAnimationSpeedFactor = 1.0 - 0.4 * initialSpeedScalingEaseIO;
  globalTime += globalAnimSpeed * initAnimationSpeedFactor * deltaTime;

  // Build nodes once
  if (nodes.length === 0) {
    const spacingHeight = height * 1.3;
    const spacingWidth = width * 1.3;
    treeStructData.layout.forEach((lay, index) => {
      new Node(
        spacingWidth * lay[0] - spacingWidth / 2,
        20 + spacingHeight * lay[1] - spacingHeight / 2,
        1 - treeStructData.relSize[index],
        index
      );
    });
    treeStructData.edges.forEach((ind) => {
      nodes[ind[0]].child(nodes[ind[1]]);
    });
    nodes.forEach((n) => {
      if (n.isRootNode) n.startAnimation();
    });
    nodes[0].selected = true;
    setNodeDetails(0);

    // Add event listener for the copy button after nodes are built
    const copyButton = document.getElementById('copy-data-btn');
    if (copyButton) {
      copyButton.addEventListener('click', () => {
        const selectedNode = nodes.find(n => n.selected);
        if (selectedNode && treeStructData && treeStructData.node_data_list) {
          const nodeData = treeStructData.node_data_list[selectedNode.treeInd];
          if (nodeData) {
            // Convert the node data to a pretty-printed JSON string
            const jsonString = JSON.stringify(nodeData, null, 2);
            copyToClipboard(jsonString);
          } else {
            console.error('Could not find data for selected node index:', selectedNode.treeInd);
          }
        } else {
          console.log('No node selected or node data unavailable.');
        }
      });
    }
  }

  // Possibly auto-select the largest node that's static:
  const staticNodes = nodes.filter((n) => n.isStatic || n.animationProgress >= 0.7);
  if (staticNodes.length > 0) {
    const largestNode = staticNodes.reduce((prev, current) =>
      prev.relSize > current.relSize ? prev : current
    );
    if (!manualSelection) {
      if (!largestNode.selected) {
        setNodeDetails(largestNode.treeInd);
      }
      staticNodes.forEach((node) => {
        node.selected = node === largestNode;
      });
    }
  }

  background(bgCol);

  // main transform for the "world"
  translate(width / 2 + offsetX, height / 2 + offsetY);
  scale(zoomFactor * baseScaleFactor);

  // Draw edges first
  edges.forEach((e) => e.render());

  // Then draw nodes
  nodes.forEach((n) => n.render());

  
  const higher    = treeStats.childHigherThanParent();
  const lower    = { hit: higher.total - higher.hit, total: higher.total, pct: 1 - higher.pct };
  const diffAgg   = treeStats.scoreChangeAggregates();
  const bugInfo   = treeStats.bugStats();
  const depthInfo = treeStats.debugDepthStats();
  const statsBox  = document.getElementById("tree-stats");

  if (statsBox){
    statsBox.innerHTML = `
      <section>
        <h4>Child &gt; Parent</h4>
        <p class="metric">
          <span class="val">${(higher.pct*100).toFixed(1)} %</span>
          <small>(${higher.hit}/${higher.total})</small>
        </p>
      </section>

      <section>
        <h4>Child &lt; Parent</h4>
        <p class="metric">
          <span class="val">${(lower.pct*100).toFixed(1)} %</span>
          <small>(${lower.hit}/${lower.total})</small>
        </p>
      </section>
  
      <section>
        <h4>Δ Score (child – parent)</h4>
        <p class="metric">
          mean <span class="val">${diffAgg.mean.toFixed(2)}</span>,
          σ <span class="val">${diffAgg.stdev.toFixed(2)}</span>, 
          median <span class="val">${diffAgg.median.toFixed(2)}</span><br>
          max ↑ <span class="val">${diffAgg.maxImprovement.toFixed(2)}</span>,
          max ↓ <span class="val">${diffAgg.maxDrop.toFixed(2)}</span>
        </p>
      </section>
  
      <section>
        <h4>Buggy‑node stats</h4>
        <p class="metric">
          bugs <span class="val">${bugInfo.bugCount}/${bugInfo.total}</span>
          (<span class="val">${(bugInfo.rate*100).toFixed(1)} %</span>)<br>
          parent bug → child bug
          <span class="val">${(bugInfo.propagation*100).toFixed(1)} %</span><br>
          📈 bug → clean
          <span class="val">${(bugInfo.fixRate*100).toFixed(1)} %</span><br>
          buggy‑sibling clusters (≥2)
          <span class="val">${bugInfo.siblingClusters}</span>
        </p>
      </section>
  
      <section>
        <h4>Debug‑depth chains</h4>
        <p class="metric">
          avg <span class="val">${depthInfo.avg.toFixed(2)}</span>
          (from ${depthInfo.total})<br>
          depth 10 chains:
          <span class="val">${depthInfo.depth10}</span>
        </p>
      </section>
    `;
  }

};

// Only zoom if mouse is in canvas
function mouseWheel(event) {
  if (mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height) {
    zoomFactor -= event.delta * 0.001;
    zoomFactor = constrain(zoomFactor, 0.01, 10);
    return false;
  }
}

function mousePressed() {
  // Convert mouse to world coords for dragging
  const [mxWorld, myWorld] = screenToWorld(mouseX, mouseY);
  draggingNode = null;

  // Check if we clicked on a node
  for (let i = nodes.length - 1; i >= 0; i--) {
    let n = nodes[i];
    const scaledRadius = (n.renderSize / (zoomFactor * baseScaleFactor));
    if (dist(mxWorld, myWorld, n.x, n.y) < scaledRadius) {
      // We clicked this node
      nodes.forEach((node) => (node.selected = false));
      n.selected = true;
      setNodeDetails(n.treeInd);
      manualSelection = true;

      // We'll drag the node in world coords
      draggingNode = n;
      nodeDragOffsetX = n.x - mxWorld;
      nodeDragOffsetY = n.y - myWorld;

      isPanning = false;
      return;
    }
  }

  // If no node was clicked, we'll start panning
  isPanning = true;
}

function mouseDragged() {
  if (mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height) {
    if (draggingNode) {
      // Move the node in world coords
      const [mxWorld, myWorld] = screenToWorld(mouseX, mouseY);
      draggingNode.x = mxWorld + nodeDragOffsetX;
      draggingNode.y = myWorld + nodeDragOffsetY;
    } else if (isPanning) {
      // Pan offsets in screen space
      offsetX += movedX;
      offsetY += movedY;
    }
  }
}

function mouseReleased() {
  draggingNode = null;
  isPanning = false;
}

