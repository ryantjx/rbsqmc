1. `run_model_unbiased_colab.sh`
   1. create folder: `rbpf/outsputs/smoothing_unbiased_gpu_DDMMYYYY_HHMM`
   2. setup colab environment
      1. check RAM, GPU Model and print the parameters
   3. `run_smoothing_gpu.py`
      1. Check required packages, download ONLY if required.
      2. run smoothing
      3. download results from gpu
         1. `params_unbiased.json`
         2. `optimization_summary.json`
   4. stops server
   5. download run parameters `run_config.json`
   6. generate images locally from GPU outputs
      1. generates images locally
      2. optimization_logZ_curve.png
   7. create folder : `rbpf/outsputs/smoothing_unbiased_gpu_DDMMYYYY_HHMM/filtered`
   8. run `model.py` locally - generate filtered states. takes in a parameter path and a run config path
      1. download results in `filtered`
         1. `filtered_states.npz`
         5. `timeseries_states.json`
      2. generate output images
         1. `final_rankings.png`
         2. `timeseries_states.png`
         3. `top_strengths.png`
   9.  create folder : `rbpf/outsputs/smoothing_unbiased_gpu_DDMMYYYY_HHMM/predict`
   10. run `predict.py` - runs sequential prediction using the model's latest model filtered states. takes `predictions.json` and `filtered_states.npz`
      1. Generate prediction in `/predict`
         1. `predictions.json`
         2. `post_prediction_filter_rankings.json` - final rankings after running filter
      2. Generate images of all games in `rbpf/outsputs/smoothing_unbiased_gpu_DDMMYYYY_HHMM/predict/prediction_plots`
   11. download logs in `rbpf/outsputs/smoothing_unbiased_gpu_DDMMYYYY_HHMM`
   12. stop colab session