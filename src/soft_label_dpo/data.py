"""Turn the four human votes on each MultiPref pair into soft labels.

Each pair in the "default" configuration of MultiPref has four votes on which of its two answers, A or B,
is better. Two of the votes come from crowdworkers and two from experts. The "overall_pref" of each vote
is coded as A = 1, Tie (T) = 0.5 or B = 0, with "slightly better" and "clearly better" counting the same.

From these numbers, each pair gets two values:

- p, the share of votes for A, with a tie counting as half a vote for each answer. It is the mean of the
  numbers, so AAAB (three votes for A, one for B) gives p = 0.75. p = 0.5 means there is no majority.

  p(AAAB) = (1+1+1+0)/4 = 0.75
  p(AABB) = (1+1+0+0)/4 = 0.5
  p(TTTT) = (0.5+0.5+0.5+0.5)/4 = 0.5
  p(BBBT) = (0+0+0+0.5)/4 = 0.125

- d, how much the annotators disagree. It is the population variance of the numbers, so it runs from 0,
  when all votes are the same, to 0.25, when half are for A and half are for B. TTTT and AABB both give
  p = 0.5, but TTTT gives d = 0 and AABB gives d = 0.25.

  d(AAAB) = ((1-0.75)^2 + (1-0.75)^2 + (1-0.75)^2 + (0-0.75)^2)/4 = 0.1875
  d(AABB) = ((1-0.5)^2 + (1-0.5)^2 + (0-0.5)^2 + (0-0.5)^2)/4 = 0.25
  d(TTTT) = ((0.5-0.5)^2 + (0.5-0.5)^2 + (0.5-0.5)^2 + (0.5-0.5)^2)/4 = 0
  d(BBBT) = ((0-0.125)^2 + (0-0.125)^2 + (0-0.125)^2 + (0.5-0.125)^2)/4 = 0.109375

Each pair also gets an agreement bucket (strong, moderate or split) from p and a vote pattern, which is
its votes as sorted letters, such as AABT. The functions that take a list of votes accept any number of
them, so they also work on the crowdworker or expert votes alone.
"""
