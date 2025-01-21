import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle numpy scalar types (e.g., np.int64, np.float64)
        if isinstance(obj, np.generic):
            return obj.item()  # Convert to native Python type (e.g., int, float)

        # Handle numpy arr` ays (convert to a list)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        # Fallback to the default method for other types
        return super().default(obj)


def merge_json(records, extant_records, dict_fields={}, fields_to_remove=()):
    for id, data in records.items():

        for fieldname, label in dict_fields.items():

            data[fieldname] = {
                label: data[fieldname],
            }

            try:
                extant_data_field = extant_records[id][fieldname]
            except KeyError:
                extant_data_field = {}

            if isinstance(extant_data_field, dict):
                # is a dict; overwrite this field if it exists but don't touch others
                data[fieldname] = {**extant_data_field, **data[fieldname]}
            # TODO: deal with known, non-dict value, creating a dict (eg, string)
            else:
                raise TypeError(f"Extant field {fieldname} must be a dict.")

        # remove some keys - not the same across plots
        for x in fields_to_remove:
            try:
                data.pop(x)
            except KeyError:
                pass

        # overwrite extant records
        extant_records[id] = data

    return extant_records
