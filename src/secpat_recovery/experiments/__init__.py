"""The paper's experiments on its dataset, run through the `secpat` command.

- ``secpat evaluate`` (``runner``) scores the dataset's stored recoveries again.
- ``secpat interaction`` (``runner``) scores the dataset's stored recoveries,
  deriving each one's interaction order from its role mapping first.
- ``secpat recover --models`` (``runner``) runs the pipeline, the baseline or an
  ablation arm with each model, then scores.
- ``secpat report`` (``report``) rebuilds the output folder's ``index.html``.

``config.toml`` lists the models; ``dataset`` knows the dataset's layout.
"""
