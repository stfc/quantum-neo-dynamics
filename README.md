A repository to accompany the paper "Approximate quantum circuit compilation for proton-transfer kinetics on quantum processors". Here we provide a set of circuits designed to model dynamics of proton transfer in malonaldehyde using the [Nuclear Electronic Orbital theory](https://doi.org/10.1063/1.1494980). We also provide the results generated from our simulations that are described in the paper.

### 🛠️ Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/stfc/quantum-neo-dynamics.git
   cd quantum-neo-dynamics
   ```
2. Create a virtual environment using Python 3.11:
   ```
   python3.11 -m venv venv
   ```
3. Activate the virtual environment:
   -On Linux/macOS
   ```
   source venv/bin/activate
   ```
   -On Windows
   ```
   venv\Scripts\activate
   ```
4. Install the required dependencies
   ```
   pip install -r requirements.txt
   ```
   
### Executing programme

* To run the main programme, make sure you have Python installed and the required dependencies set up (seethe Installation section). Then, to perform statevector simulations, execute qneo.py as
```
python qneo.py -m <method> -s <system> -a <approximation> -s <state>
```
* Additional arguments required for noisy simulations are specified within the script.
* To save custom noise models, execute save-noise-models.py as
```
python save-noise-models.py -y <year> -m <month> -d <day> -H <hour> -M <minutes> -S <seconds>
```

## Authors

- [@dilhanm](https://github.com/DilhanM)
- [@edoaltamura](https://github.com/edoaltamura)
- [@GeorgePennington](https://github.com/GeorgePennington)
- [@bjader](https://github.com/bjader)

## Version History

* 0.1
    * Initial Release

## License

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Acknowledgments
