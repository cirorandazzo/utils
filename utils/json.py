import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder class to handle numpy data types when serializing to JSON.

    This class extends the `json.JSONEncoder` and provides custom handling for numpy
    scalar types and arrays, converting them to standard Python types (e.g., int, float, list).
    """

    def default(self, obj):
        """
        Override the default method to handle numpy types.

        Parameters:
        ----------
        obj : any
            The object to be serialized.

        Returns:
        -------
        serializable : any
            The object converted into a serializable form (native Python types).
        """
        # Handle numpy scalar types (e.g., np.int64, np.float64)
        if isinstance(obj, np.generic):
            return obj.item()  # Convert to native Python type (e.g., int, float)

        # Handle numpy arrays (convert to a list)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        # Fallback to the default method for other types (e.g., non-numpy objects)
        return super().default(obj)


def merge_json(
    records,
    extant_records,
    dict_fields={},
    fields_to_remove=(),
    keep_extant_fields=True,
):
    """
    Merges new records into an existing set of records.

    This function updates `extant_records` by iterating through the new `records` with the following behaviors:
    - Fields in `dict_fields` will be stored as a sub-dictionary, with the corresponding value as the key of the sub-dictionary for each record.
    - Fields in both `records` and `extant_records` will be updated to `records`.
    - Fields in `fields_to_remove` are removed.

    Parameters:
    ----------
    records : dict
        A dictionary where each key represents a record ID and the value is the data associated with that ID.

    extant_records : dict
        A dictionary of existing records that will be updated with new information from `records`.

    dict_fields : dict, optional, default={}
        A dictionary mapping field names to labels. For each field in `dict_fields`, the corresponding
        value in `records` will be wrapped in a dictionary with the label as the key.

    fields_to_remove : tuple, optional, default=()
        A tuple of field names that should be removed from all records.

    keep_extant_fields : bool, optional, default=True
        Behavior on fields in extant_records but not in records. If True, keeps these fields. If False, deletes these fields.

    Returns:
    -------
    dict
        The updated `extant_records`, which now includes merged data from `records`.
    """
    # Iterate over each record in the new 'records' dictionary
    for id, data in records.items():

        # Process fields specified in dict_fields for each record
        for fieldname, label in dict_fields.items():
            # Wrap the field's value in a dictionary with the provided label
            data[fieldname] = {
                label: data[fieldname],
            }

            # Try to get the corresponding field from the extant records, if it exists
            try:
                extant_data_field = extant_records[id][fieldname]
            except KeyError:
                extant_data_field = {}  # If it doesn't exist, initialize as an empty dict

            if isinstance(extant_data_field, dict):
                # If the existing field is a dict, merge the new field with the extant field
                data[fieldname] = {**extant_data_field, **data[fieldname]}
            else:
                # Raise an error if the extant field is not a dict (future work: handle non-dict types)
                raise TypeError(f"Extant field {fieldname} must be a dict.")

        # Remove unwanted fields (those in the fields_to_remove list)
        for x in fields_to_remove:
            try:
                data.pop(x)
            except KeyError:
                pass  # Ignore if the field doesn't exist in the record

        if keep_extant_fields:  # overwrite changed fields, but keep fields not in data
            extant_records[id] = {**extant_records[id], **data}
            data[fieldname] = {**extant_data_field, **data[fieldname]}
        else: # Overwrite the record in extant_records with updated data
            extant_records[id] = data

    return extant_records
