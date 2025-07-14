Approximate quantum circuit compilation for proton-transfer kinetics on quantum processors.

## Getting Started

### Dependencies

* 
* 

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
## Help


```

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



## Acknowledgments
