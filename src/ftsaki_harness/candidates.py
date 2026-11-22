"""Durable single-forecast boundary: begin must finish before advancing the source."""
from .forecast import Forecast

class Candidates:
    def __init__(self, records):
        self.records = records

    def begin(self, forecast):
        key = "forecast/" + forecast.candidate
        existing = self.records.get(key)
        if existing is not None:
            if existing["frozen_hash"] != forecast.frozen_hash:
                raise ValueError("Candidate already forecast once")
            return Forecast.restore(existing)
        self.records.commit({key: forecast.dump()})
        return forecast

    def advance(self, candidate, samples):
        result = self.records.get("forecast-result/" + candidate)
        if result is not None:
            return result
        initial = self.records.get("forecast/" + candidate)
        if initial is None:
            raise ValueError("Forecast must be persisted before target arrival")
        steps = [(k,v) for k,v in self.records.export().items() if k.startswith("target/" + candidate + "/")]
        forecast = Forecast.restore(initial)
        outcome = None
        for _, step in sorted(steps):
            outcome = forecast.append(**step)
            if outcome:
                break
        for sample in samples:
            if forecast.closed:
                break
            key = "target/" + candidate + "/" + f"{len(forecast.ids):03d}"
            if sample["sample_id"] in forecast.ids:
                continue
            # Store the original source position, including invalid rows, before scoring.
            self.records.commit({key: sample})
            outcome = forecast.append(**sample)
        if outcome:
            self.records.commit({"forecast-result/" + candidate: outcome})
        return outcome

    def end(self, candidate):
        result = self.records.get("forecast-result/" + candidate)
        if result is not None:
            return result
        initial = self.records.get("forecast/" + candidate)
        if initial is None:
            raise ValueError("Unknown candidate")
        forecast = Forecast.restore(initial)
        for key, step in sorted(self.records.export().items()):
            if key.startswith("target/" + candidate + "/"):
                outcome = forecast.append(**step)
                if outcome:
                    self.records.commit({"forecast-result/" + candidate: outcome})
                    return outcome
        result = forecast.end()
        self.records.commit({"forecast-result/" + candidate: result})
        return result
