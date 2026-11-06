# Coding style to follow - what repo developer find easy to understand:
- Functional approach, take f.ex. look at StateNetworkTrainer main loop, when function args and returns are clearly stated.
- Tensor methods - avoid creating intermediate object that don't have clear name and purpose, f.ex. obj2 = obj.T
- Tensor methods - as little reshaping/repeating stacking as possible, as it is hard to track.
- Function logic should go without stops, if some helper needs to be calculated, make this a task for helper function.