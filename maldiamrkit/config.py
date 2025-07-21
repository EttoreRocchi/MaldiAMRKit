from dataclasses import dataclass

@dataclass()
class PreprocessingSettings:

    trim_from: int = 2_000
    trim_to:   int = 20_000

    savgol_window: int = 20
    savgol_poly:   int = 2
    baseline_half_window: int = 40

    def as_dict(self) -> dict:
        return self.__dict__.copy()
