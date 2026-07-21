# Code Performance Advisor - Architecture Documentation

**Version**: 2.0
**Date**: 2026-02-25
**Status**: Design Proposal

---

## 📚 Documentation Suite

This directory contains the complete architectural design for Code Performance Advisor v2.0.

### Core Documents

1. **[ARCHITECTURE.md](./ARCHITECTURE.md)** ⭐ **(Start Here)**
   - System overview and design philosophy
   - 5-layer architecture (Interface → Orchestration → Analysis → Execution → Knowledge)
   - Module organization and responsibilities
   - Migration plan (5-week roadmap)
   - Design principles and trade-offs

2. **[DIAGRAMS.md](./DIAGRAMS.md)**
   - Visual representations of system design
   - Layer diagrams, data flow, state transitions
   - Module dependency graph
   - Sequence diagrams for key workflows
   - Error handling flows

3. **[API_AND_MIGRATION.md](./API_AND_MIGRATION.md)**
   - Complete API specifications for all core modules
   - Data model definitions with type hints
   - Detailed migration guide (week-by-week)
   - Developer onboarding instructions
   - Code style guidelines

---

## 🎯 Quick Navigation

### For Project Managers
- Read: **ARCHITECTURE.md** Section 1-2 (Overview, Layered Architecture)
- Focus: Migration Plan (Section 7), Success Metrics (Section 9)

### For Architects/Tech Leads
- Read: **All three documents**
- Focus: ARCHITECTURE.md Section 3-6 (Layer designs, Module organization)
- Review: DIAGRAMS.md (System interactions)

### For Developers
- Read: **API_AND_MIGRATION.md** Part I (API Specification)
- Focus: Data models (Section 1), Module APIs (Section 2)
- Reference: DIAGRAMS.md Section 6 (Module dependency graph)

### For QA/Test Engineers
- Read: **ARCHITECTURE.md** Section 9 (Success Metrics)
- Focus: API_AND_MIGRATION.md Part II Section 6 (Migration Validation)

---

## 🔑 Key Highlights

### Design Principles

1. **Compiler-like Precision**: Clear phases, deterministic execution
2. **Knowledge Evolution**: Every validated optimization becomes reusable knowledge
3. **Automation First**: Automate deterministic tasks, use LLM for pattern recognition only
4. **Verifiable Quality**: No suggestion is "done" until validated
5. **Progressive Disclosure**: Start simple, escalate only when needed

### Major Improvements Over v1.0

| Aspect | v1.0 (Current) | v2.0 (Proposed) |
|--------|----------------|-----------------|
| **Entry Point** | Multiple scripts | Single `advisor` CLI |
| **Automation** | Phase 0 only | End-to-end (Phase 0-4) |
| **State Management** | None | Persistent, resumable |
| **Modularity** | Scattered | 5-layer architecture |
| **Code Reuse** | Duplication | Shared core modules |
| **Testing** | Limited | Full unit + integration |
| **Documentation** | Inconsistent | Complete API specs |

### Architecture at a Glance

```
User → CLI → Advisor → Router → Analysis → Execution → Validation → Knowledge
                                    ↓           ↓          ↓            ↓
                              Rule Match   Transform   Verify      Capture
                              Suggest      Build       Compare     Update
                              Evidence     Profile     Goal        Rules
```

---

## 📊 Implementation Status

### Completed (v1.0)
- ✅ Rule library structure
- ✅ Tag-based matching (Phase 0)
- ✅ Basic CLI (score command)
- ✅ Goal configuration support
- ✅ Suggestion template framework

### In Progress
- 🚧 Suggestion generation automation (35% complete)
- 🚧 Evidence extraction
- 🚧 Code pattern matching

### Planned (v2.0 Migration)
- ⏳ 5-layer architecture implementation
- ⏳ End-to-end optimization loop
- ⏳ State management and persistence
- ⏳ Knowledge capture automation
- ⏳ Performance model training

---

## 🚀 Getting Started with v2.0

### Phase A (Week 1) - Foundation
**Goal**: Set up new structure without breaking existing functionality

**Key Tasks**:
1. Create `core/` directory structure
2. Implement core data models
3. Create `advisor.py` main entry point
4. Implement `advisor analyze` command (stub)

**Success**: `advisor analyze fastgelu` works alongside old scripts

### Phase B (Week 2) - Suggestion Pipeline
**Goal**: Complete suggestion generation automation

**Key Tasks**:
1. Implement `core/analysis/suggestion_generator.py`
2. Implement `core/analysis/evidence_extractor.py`
3. Implement `advisor suggest` command
4. Integration test with fastgelu

**Success**: `advisor suggest fastgelu` generates high-quality suggestions

### Phase C-E (Week 3-5)
See **ARCHITECTURE.md Section 7** for complete roadmap.

---

## 📖 Document Change Log

### 2026-02-25 - Initial v2.0 Design
- Created ARCHITECTURE.md (complete system design)
- Created DIAGRAMS.md (visual representations)
- Created API_AND_MIGRATION.md (technical specifications)
- Established 5-layer architecture
- Defined migration plan (5 weeks)

---

## 🤝 Contributing

### Feedback on Architecture
- Review documents and provide feedback via GitHub issues
- Tag issues with `architecture-review`
- Focus areas: scalability, testability, maintainability

### Suggesting Improvements
- Propose alternative designs with rationale
- Consider: complexity vs. benefit trade-offs
- Reference specific sections (e.g., "ARCHITECTURE.md Section 2.3")

### Questions
- Use GitHub Discussions for questions
- Tag with `architecture` or `migration`

---

## 📞 Contact

- **Architecture Lead**: [To be assigned]
- **Migration Lead**: [To be assigned]
- **Project Channel**: #code-performance-advisor

---

## 📎 Related Resources

### Internal
- `../../SKILL.md` - Skill metadata for Claude Code
- `../standards/` - Performance thresholds, CSV formats
- `../../tests/` - Test suite

### External
- CANN Toolchain Documentation
- AscendC Programming Guide
- Performance Optimization Best Practices

---

## ⚠️ Important Notes

### Backward Compatibility
- v2.0 migration maintains backward compatibility during transition
- Old scripts remain functional with deprecation warnings
- Workspace data format is preserved

### Breaking Changes (Post-Migration)
- CLI commands change (from `python cli.py` to `advisor`)
- Import paths change (from `scripts.*` to `core.*`)
- Configuration file format (new YAML schema)

**Migration Timeline**: 5 weeks
**Rollback Plan**: Available (see API_AND_MIGRATION.md Section 7)

---

**Last Updated**: 2026-02-25
**Next Review**: After Phase A completion (Week 1)
