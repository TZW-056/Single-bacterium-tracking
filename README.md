# Install

Clone this repository and install requirements.txt

```bash
$ git clone https://github.com/TZW-056/Single-bacterium-tracking.git
$ cd Single-bacterium-tracking
$ pip install -r requirements.txt
```

# Usage

The following is the structure of this repository. 

```bash
.
│  .gitignore
│  README.md
│  requirements.txt  
│
├─data
│  ├── Test-B02.avi  # 20% brighter than the original video
│  ├── Test-R2.avi   # double size of the original video
│  └── Test.avi      # the original video
│      
├─outputs
│   ├── B00.png    # trajectory figure according to Test_yxt.xlsx
│   ├── B20.png    # trajectory figure according to Test-B02_yxt.xlsx
│   ├── Test-B02_yxt.xlsx # trajectory data obtained by processing the Test-B02.avi 
│   └── Test_yxt.xlsx  # trajectory data obtained by processing the Test.avi
│      
└─src
   ├── Plot.py         # visualization trajectory data
   ├── SBTAnalyzer.py  # extraction of bacterial trajectories
   └── utils.py        # some needed functiones
```

The input video formats that  `SBTAnalyzer.py` supports are `.avi` `.mp4` and the outputs will be stored in the `./outputs` folder with the suffix name of `.xlsx`.  What's more, `Plot.py` can produce the trajectory figure according to the output of  `SBTAnalyzer.py` 

NOTE: `Test-B02.avi` and `Test-R2.avi` were obtained by `ffmpeg`.

# License

[Apache Licence 2.0](http://www.apache.org/licenses/LICENSE-2.0)
