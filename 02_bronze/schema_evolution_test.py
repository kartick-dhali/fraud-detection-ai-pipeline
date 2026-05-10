"""Document and validate the four schema-evolution scenarios used by Bronze ingestion.

These are lightweight unit tests because the repository does not ship a larger pre-existing
test framework. The goal is to keep schema behavior explicit and easy to review.
"""

import unittest


SCENARIOS = [
    {
        "name": "additive_column",
        "sample_record": {
            "TransactionID": "tx-1",
            "TransactionDate": "2025-01-01T08:00:00",
            "Amount": 125.75,
            "DeviceID": "ios-99",
        },
        "expected_behavior": "mergeSchema adds DeviceID without replaying old files.",
        "why": "Optional new telemetry should not break ingestion.",
    },
    {
        "name": "type_drift",
        "sample_record": {
            "TransactionID": "tx-2",
            "TransactionDate": "2025-01-01T09:00:00",
            "Amount": "unknown",
        },
        "expected_behavior": "schemaHints preserves canonical type and _rescued_data captures drift.",
        "why": "A single bad amount should be inspected, not crash the stream.",
    },
    {
        "name": "sparse_optional_attribute",
        "sample_record": {
            "TransactionID": "tx-3",
            "TransactionDate": "2025-01-01T10:00:00",
            "Channel": "atm",
        },
        "expected_behavior": "Rows without Channel still load with null values.",
        "why": "Partner feeds often introduce fields gradually.",
    },
    {
        "name": "nested_payload_expansion",
        "sample_record": {
            "TransactionID": "tx-4",
            "TransactionDate": "2025-01-01T11:00:00",
            "RiskSignals": {"velocity": "high", "geo_match": False},
        },
        "expected_behavior": "Unexpected nested payload is retained in _rescued_data for later parsing.",
        "why": "Engineering can review the new payload before promoting it downstream.",
    },
]


class SchemaEvolutionTests(unittest.TestCase):
    def test_all_four_scenarios_are_present(self) -> None:
        self.assertEqual(len(SCENARIOS), 4)

    def test_each_scenario_has_explanation(self) -> None:
        for scenario in SCENARIOS:
            self.assertTrue(scenario["expected_behavior"])
            self.assertTrue(scenario["why"])

    def test_type_drift_mentions_rescue(self) -> None:
        type_drift = next(s for s in SCENARIOS if s["name"] == "type_drift")
        self.assertIn("_rescued_data", type_drift["expected_behavior"])

    def test_additive_column_mentions_merge_schema(self) -> None:
        additive = next(s for s in SCENARIOS if s["name"] == "additive_column")
        self.assertIn("mergeSchema", additive["expected_behavior"])


if __name__ == "__main__":
    unittest.main()
