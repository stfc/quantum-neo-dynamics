import argparse
from datetime import datetime, timezone, UTC
import pickle, json
from quantum_neo_dynamics.paths import TARGETS_DIR, BACKENDS_DIR
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.providers.models.backendproperties import BackendProperties
from qiskit_ibm_runtime.models.backend_properties import BackendProperties as IBMBackendProperties

def serialize_datetime(o):
    """Custom JSON serializer for datetime objects."""
    if isinstance(o, (datetime,)):
        return o.isoformat()
    raise TypeError(f"Type {type(o)} not serializable")

def main():
    parser = argparse.ArgumentParser(description="Fetch and save backend properties and target history.")
    parser.add_argument("-y", "--year", type=int, required=True, help="Year (e.g., 2024)")
    parser.add_argument("-m", "--month", type=int, required=True, help="Month (1-12)")
    parser.add_argument("-d", "--day", type=int, required=True, help="Day (1-31)")
    parser.add_argument("-H", "--hour", type=int, required=True, help="Hour (0-23)")
    parser.add_argument("-M", "--minute", type=int, required=True, help="Minute (0-59)")
    parser.add_argument("-S", "--second", type=int, required=True, help="Second (0-59)")

    args = parser.parse_args()

    target_time = datetime(args.year, args.month, args.day, args.hour, args.minute, args.second, tzinfo=timezone.utc)

    device = "ibm_fez"
    service = QiskitRuntimeService()
    backend = service.backend(device)

    props = backend.properties(datetime=target_time)
    target_history = backend.target_history(datetime=target_time)
    props_dict = props.to_dict()
    props_converted = BackendProperties.from_dict(
        IBMBackendProperties.from_dict(props_dict).to_dict()
    )

    # Save target
    target_time_str = target_time.astimezone(UTC).isoformat()
    target_filename = f"{TARGETS_DIR}/{device}_{target_time_str}.target.pkl"
    with open(target_filename, "wb") as target_file:
        pickle.dump(target_history, target_file)

    # Save props
    props_filename = f"{BACKENDS_DIR}/{device}_{target_time_str}.properties.json"
    with open(props_filename, "w") as json_file:
        json.dump(props_converted.to_dict(), json_file, default=serialize_datetime)

if __name__ == "__main__":
    main()
