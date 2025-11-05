# Technical Writing Style Guide

*For GitHub Portfolio Articles*

---

## Purpose & Context

**For Claude (the AI):** This guide provides a basis for evaluating NEW articles and making useful suggestions. When reviewing articles:

1. **Check against these patterns** - Identify where the article aligns or diverges from established style
2. **Note what's working well** - Call out effective patterns the author used
3. **Suggest specific improvements** - Point to sections that could be strengthened with examples
4. **Flag missing elements** - TOC, decision frameworks, error handling, etc.
5. **Evaluate the "wish I had this" test** - Would this have helped someone learning this?
6. **Check for personal voice** - Is this generic tutorial content or hard-won expertise?
7. **Verify the ladder** - Can readers at different levels find their entry point?

**Guiding Principle:** Write the article I WISH someone had written when I was learning this stuff. Not the article that exists. Not the sterile tutorial. The one that would have actually helped past-me get unstuck and move forward.

**For Readers:** If you get confused or stuck while following one of these articles, paste the relevant section into your preferred LLM (ChatGPT, Claude, etc.) and ask it to guide you through the tricky bits. These articles are written to be LLM-friendly for exactly this reason—they provide clear context and examples that LLMs can help you adapt to your specific situation.

---

## Core Principles

### Code Presentation

**SHORT CODE BLOCKS**
- Keep examples focused on one concept
- Break complex configs into digestible chunks
- If it's more than 20 lines, consider splitting it

**Show CODE, then explain**
```bash
# Example code here
command --flag value
```
*Then* add your explanation. Some readers get it immediately and want to move on. Don't clutter their first read.

**Multiple passes:**
1. Clean code block (no comments)
2. Same code with annotations
3. Variations or edge cases

### Content Structure

**SHOW first, editorialize second**
- Present the technical content cleanly
- Then add your take, experience, opinions
- Value = your expertise + personality, not pure LLM output

**Use cases drive everything**
- Don't explain features in isolation
- Show WHY someone would use this
- Real scenarios > abstract capabilities

**Concrete examples > abstract descriptions**
- Bad: "This improves productivity"
- Good: "Type `cd /u/l/b<tab>` → expands to `/usr/local/bin`"

**Ladder of complexity**
- Start with simplest useful version
- Show what it gets you
- Provide path to advanced usage
- Make readers feel "I could do this"

### Comparisons and Trade-offs

**Side-by-side comparisons**
```
**Bash:**
<command>

**Zsh:**
<different approach>
```

**Acknowledge trade-offs honestly**
- Not every solution fits every problem
- Show when NOT to use something
- Present counter-arguments fairly

**"What this gets you" / "What it doesn't do"**
- Set realistic expectations
- Prevent feature bloat
- Help readers make informed decisions

---

## Article Structure

### Opening
- Hook with context (why this matters NOW)
- Personal stake if relevant ("I like bash... So there's resistance")
- Question or clear value proposition

### Table of Contents
- Use for articles > 150 lines
- Link to all major sections
- Include subsections for complex topics

### Body Patterns

**Present anti-patterns early**
- "Skip Oh My Zsh" comes BEFORE the alternative
- Saves reader time
- Establishes credibility through selectivity

**Multiple paths to success**
- Quick path for experienced users
- Detailed walkthrough for learners
- "What I actually did" for pragmatists

**Decision frameworks at the end**
- "Consider X if..."
- "Stay with Y if..."
- Help readers self-select

### Voice and Tone

**Opinionated but self-aware**
- "features you probably don't need. Ok, *fine*—that *I* don't need"
- Strong views, loosely held
- Acknowledge subjectivity

**Conversational but precise**
- Metaphors welcome ("hammer-shopping", "Keystroke Golf")
- Casual language okay, profanity removed for publication
- Technical accuracy is non-negotiable

**Respect reader's time**
- "This is a short one"
- Front-load most valuable info
- Provide escape hatches ("if bash works, keep it")

**Direct and pragmatic**
- No unnecessary superlatives
- Acknowledge when "good enough" is fine
- Value function over perfection

---

## Technical Details

### Code Formatting

**Inline code:**
- Triple backticks (```command```) for commands to be typed literally
- Single backticks for variables, short references

**Code blocks:**
- Always specify language for syntax highlighting
- Include platform-specific notes in comments when relevant
- Show actual output when it matters

**Comments in code:**
- First pass: NO comments (clean example)
- Second pass: Explain WHY, not WHAT
- Focus on non-obvious implications

### Platform Specificity

**Call out platform differences**
```bash
sudo apt install zsh  # or brew install zsh on Mac
```

**Test on primary platforms**
- macOS and Linux minimum
- Note Windows/WSL differences if relevant

### Links and References

**Link to official docs for:**
- Deep dives you won't cover
- API references
- Version-specific details

**Don't link for:**
- Basic concepts your audience knows
- Things you're explaining fully
- Common commands

---

## Quality Checks

### Before Publishing

- [ ] All code examples tested and work
- [ ] Platform-specific instructions accurate
- [ ] TOC matches actual sections
- [ ] No profanity (even mild)
- [ ] Trade-offs acknowledged
- [ ] Decision framework included
- [ ] "What you actually get" is clear
- [ ] Technical accuracy verified
- [ ] Personal voice present but not overwhelming
- [ ] Time estimates included where relevant
- [ ] Version/dependency requirements stated
- [ ] Common errors and fixes addressed
- [ ] Security implications called out if relevant
- [ ] Non-obvious structure justified
- [ ] Date added to time-sensitive content

### Voice Check

Ask: "Does this sound like a skilled engineer sharing hard-won knowledge?"
Not: "Does this sound like a tutorial bot?"

### Value Check

Ask: "What's here that ChatGPT couldn't generate?"
- Your experience
- Your judgment calls
- Your war stories
- Your "I actually did this" moments

---

## Article Types

### Tool Comparisons
- Present both fairly
- Lead with use cases
- Decision framework mandatory
- Counter-case for balance

### How-To Guides
- Start with simplest working version
- Layer complexity progressively
- Multiple approaches if useful
- "What I actually did" section

### Configuration Guides
- Minimal version first
- Explain each addition
- Full example at end
- "What this gets you" section

---

## Anti-Patterns to Avoid

**Don't:**
- Show complex code without building up to it
- Explain before showing
- Present only one way to do things
- Hide the trade-offs
- Write in pure tutorial voice
- Skip the "why bother" section
- Assume universal use case
- Leave out escape hatches

**Do:**
- Show → Explain → Expand
- Give readers permission to skip
- Acknowledge when defaults are fine
- Share what you actually use
- Make it feel achievable
- Provide the ladder

---

## Additional Patterns to Consider

### Error Handling & Troubleshooting

**Show common failure modes**
```bash
$ command
error: something went wrong
# Fix: do this instead
```

**Include the "it didn't work" section**
- Real error messages readers will encounter
- Platform-specific gotchas
- Quick fixes for common issues

**Saves reader time** by preventing "is it just me?" moments

### Version & Dependency Callouts

**Call out requirements explicitly**
```bash
# Requires zsh 5.0+
# Check with: zsh --version
```

**Why this matters:**
- Prevents frustration from outdated systems
- Helps readers self-diagnose compatibility
- Shows which features need which versions

### Structural Justifications

**Explain non-obvious ordering**
- "You can't really test drive zsh without setup..."
- Helps readers understand WHY the article flows this way
- Prevents "wait, shouldn't we do X first?" confusion

**Make intentional structure explicit** when it differs from expectations

### Visual Emphasis for Key Insights

**Pull quotes for quotable moments**
> "Best shell = the one you stop thinking about"

**Use for:**
- Core philosophy statements
- Decision-making frameworks
- Memorable takeaways

**Helps skimmers** catch the most important points

### Time Estimates & Scope Signals

**Give readers time budgets**
- "less than a minute"
- "~20 line .zshrc"
- "This is a short one"

**Helps readers decide:**
- Can I do this now?
- Should I bookmark for later?
- Is this worth my time today?

### Performance & Resource Notes

**When relevant, include:**
- Startup time impact
- Memory usage
- Disk space requirements
- Build time for compiled tools

**Example:** "No 2-second startup time" sets clear expectations

### Security Callouts

**Flag security-relevant decisions**
- "Read about `pass` if you did include sensitive information"
- API key handling
- Credential storage
- Permission implications

**Make security easy to spot** for readers skimming for risks

### Prior Art & Connections

**Connect to existing knowledge**
- "If you've used bash history search, this is similar but..."
- "Like Docker, but for..."
- "Think X meets Y"

**Reduces cognitive load** by anchoring new concepts to familiar ones

### Update Cadence & Currency

**Date time-sensitive content**
- "As of November 2025..."
- "In the current version..."
- "At the time of writing..."

**Helps readers assess:**
- Is this still relevant?
- Do I need to check for updates?
- Has the ecosystem changed?

### Next Steps & Related Topics

**End with forward momentum**
- "Once you've got this, you might want to explore..."
- "Related: [article on advanced zsh features]"
- "Common next step: setting up..."

**Provides the ladder's next rung** without overwhelming the current article

### "Why This Matters Now"

**Contextualize timing**
- "With the rise of..."
- "Now that macOS defaults to..."
- "Since [event/change/trend]..."

**Answers:** "Why am I reading this today instead of six months ago?"

---

## Questions to Ask

Before publishing, check:

1. **Would I send this to a colleague?**
2. **Does it respect their time?**
3. **Is my voice present, or is this generic?**
4. **Can someone actually DO this after reading?**
5. **Are trade-offs clear?**
6. **Is there an escape hatch?**
7. **Would this be useful in 6 months?**
8. **Is this the article I WISH existed when I was learning this?**
9. **Can a confused reader paste this into an LLM and get useful help?**
10. **Are common failure modes addressed?**

---

*"Best tool = the one you stop thinking about"*

*"Stop hammer-shopping and get back to hammering"*

Remember: Your portfolio demonstrates judgment as much as knowledge.
