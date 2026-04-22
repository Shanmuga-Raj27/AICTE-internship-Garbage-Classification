# Role
Expert Vue 3 and Tailwind CSS v4 Developer.

# Context
I am using Vite with Vue 3 and Tailwind CSS v4. My development server is crashing with the following error:
`Internal server error: Cannot apply unknown utility class 'bg-white/70'. Are you using CSS modules or similar and missing '@reference'?`

# Task
Refactor my Vue component to fix the Tailwind v4 `@apply` scoping issue.

# Requirements
1. Analyze the provided Vue component code.
2. **Primary Fix (Best Practice):** Remove the `<style scoped>` block completely. Extract all utility classes defined via `@apply` (e.g., `bg-white/70`, `rounded-lg`) and place them directly into the `class="..."` attributes of the corresponding elements in the `<template>`.
3. **Fallback Fix:** If (and only if) the `<style>` block contains complex, custom CSS that absolutely cannot be mapped to Tailwind utility classes, inject the `@reference` directive at the very top of the `<style>` block pointing to the main CSS file so Tailwind can resolve the `@apply` rules.
4. Output the complete, refactored, error-free Vue component code.

# Code to Fix:
[PASTE YOUR VUE COMPONENT CODE HERE (e.g., App.vue)]