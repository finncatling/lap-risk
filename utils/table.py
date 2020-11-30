from dataclasses import dataclass


@dataclass
class Table1Variable:
    name: str
    pretty_name: str
    in_novel_model: bool
    var_type: str  # {'continuous', 'binary', 'ordinal_multicat', 'multicat'}
    raw_data: bool = False  # If True, get from raw data, not preprocessed
    decimal_places: int = 0
