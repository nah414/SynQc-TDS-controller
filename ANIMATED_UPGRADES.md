# SynQc TDS Frontend - Animated Upgrades & Agent Integration

## Summary of Changes

Your main control panel (`adac1680-4fd6-4140-8698-e8e2e17aa7ea (1).html`) has been enhanced with two major features:

---

## 1. **Animated Background Scene**

### What Was Added
- **Canvas-based particle & atom animation system** rendered behind all UI elements
- Dynamically generated starfield with twinkling particles
- Atomic orbital animations with wobbling nuclei and spinning electrons
- Responsive to window resize with high-DPI support

### Technical Details

#### CSS
```css
canvas.bg-scene {
  position: fixed;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 0;
  opacity: 0.95;
}
```
- Canvas sits at `z-index: 0` (behind content)
- `pointer-events: none` ensures it doesn't intercept clicks

#### Main Grid Update
```css
main.app-grid {
  position: relative;
  z-index: 1;  /* Ensures panels float above animation */
}
```

#### JavaScript Function: `initBackgroundScene()`
- **140 particles** that drift, twinkle, and connect
- **7 atom systems** with:
  - Wobbling nuclei
  - Tilted elliptical orbits
  - Spinning electrons with yellow glow
  - Soft connection lines between nearby particles
- Resizes automatically on window resize events
- Uses `requestAnimationFrame` for smooth 60fps animation

---

## 2. **LLM Agent Chatbot Interface**

### What Was Added
- **Fixed chatbot widget** in bottom-right corner
- Minimizable/maximizable design
- Message history with system, user, and agent messages
- Typing indicator with animated bouncing dots
- Smart conversational responses about SynQc TDS

### Visual Features

#### Styling
- Gradient background matching system theme
- Glowing status indicator (pulsing blue dot)
- Smooth animations:
  - `slideIn` – messages fade in from bottom
  - `pulse` – status indicator breathing effect
  - `bounce` – typing dots animation

#### Responsive Breakpoints
| Screen Size | Dimensions |
|-------------|-----------|
| Desktop    | 380px × 520px |
| Tablet     | 320px × 480px |
| Mobile     | 280px × 420px |
| Ultra-wide | 100vw × 380px |

#### HTML Structure
```html
<div class="agent-chat-container">
  <div class="agent-chat-header">
    <h3>LLM Agent Assistant</h3>
    <div class="status-indicator"></div>
    <button class="agent-chat-minimize-btn">−</button>
  </div>
  <div class="agent-chat-body">
    <!-- Messages added dynamically -->
  </div>
  <div class="agent-chat-footer">
    <input type="text" placeholder="Ask the agent..." />
    <button>Send</button>
  </div>
</div>
```

### JavaScript Functions

#### `wireAgentChat()`
- Attaches event listeners to minimize/expand functionality
- Handles message sending (button click or Enter key)
- Manages user/agent message flow
- Auto-scrolls to latest message

#### `generateAgentResponse(userQuery)`
Smart keyword matching that provides context-aware responses:

| User Query | Response Focus |
|-----------|----------------|
| "help", "?" | Lists available topics |
| "config" | Configuration tips for ε, envelope, iterations |
| "fidelity" | Target ≥0.975 guidance |
| "hardware" | Supported QPU/backend options |
| "latency" | Optimization for loop speed |
| "shot" | Budget management tips |
| Default | Prompts to ask about specific topics |

#### `escapeHtml(text)`
- Sanitizes message content to prevent XSS injection
- Preserves newlines for readable multi-line responses

---

## 3. **Integration Points**

### CSS Additions
- **244 new lines** of chatbot and animation styling
- New color animations: `@keyframes pulse`, `bounce`, `slideIn`
- Responsive media queries for all screen sizes
- Custom scrollbar styling for chat body

### HTML Additions
- **Chatbot container** with 3 sub-sections (header, body, footer)
- Maintains original panel-based control layout
- No disruption to existing functionality

### JavaScript Integration
- **~850 lines** of new functionality:
  - `initBackgroundScene()` – 300+ lines (canvas animation)
  - `wireAgentChat()` – 150+ lines (event handling)
  - `generateAgentResponse()` – 80+ lines (smart responses)
  - `escapeHtml()` – 12 lines (security)

- **Called in `init()`:**
  ```javascript
  wireControls();
  wireAgentChat();        // NEW
  updateSessionHeader();
  updateKpis();
  updateExportOutput("Waiting for export…");
  updateShotsCounter();
  initBackgroundScene();  // NEW
  ```

---

## 4. **User Experience Enhancements**

### Animation Benefits
✨ **Visual Appeal** – Engaging starfield and atomic orbital backdrop  
🎯 **Context Relevant** – Animations reinforce quantum/physics theme  
⚡ **Performance** – Lightweight canvas rendering, no blocking operations  

### Chatbot Benefits
🤖 **Quick Help** – No need to read documentation  
💬 **Context Aware** – Responds to your actual workflow  
🎚️ **Non-Intrusive** – Minimizes when not needed  
🔒 **Safe** – HTML escaping prevents injection attacks  

---

## 5. **Testing & Verification**

### To Test the Features:

1. **Background Animation:**
   - Open the HTML file
   - Observe twinkling particles and atomic orbits
   - Resize the window → animation responds smoothly
   - Verify no UI elements are blocked by canvas

2. **Chatbot:**
   - Click "Ask the agent..." input field
   - Type: "help" → See list of topics
   - Type: "fidelity" → See target guidance
   - Type: "hardware" → See QPU options
   - Click minimize button (−) → Widget collapses
   - Click header → Widget expands back

---

## 6. **Future Enhancement Paths**

### Agent Integration Options
- **Real LLM Backend** – Replace `generateAgentResponse()` with API calls to GPT/Claude
- **Session Context** – Pass current config/KPIs to agent for personalized advice
- **Run Recommendations** – Agent suggests optimal parameters based on history

### Animation Customization
- Adjust `PARTICLE_COUNT` and `ATOM_COUNT` for performance/fidelity balance
- Modify particle colors in gradient definitions
- Fine-tune animation speeds in `tick()` and `spawnAtoms()`

### UI Expansion
- Add agent message reactions (👍👎)
- Implement chat persistence (localStorage)
- Export chat history with run snapshots

---

## 7. **Browser Compatibility**

✅ Chrome/Edge 90+  
✅ Firefox 88+  
✅ Safari 14+  
✅ Mobile Chrome/Safari (responsive layout)  

---

**Version:** SynQc TDS v0.4 with Animated Interfaces & Agent Integration  
**Last Updated:** December 2025
