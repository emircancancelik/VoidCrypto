"""
VoidCrypto — OrderBookAI Unit Tests
=====================================
Kritik mantık testleri:
  - Snapshot + delta senkronizasyon protokolü
  - Sequence gap tespiti
  - Price-sorted imbalance hesabı
  - Spread spike algılama
  - VWAP hesabı
  - Circuit breaker state transitions
"""
import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from order_book_ai import (
    CircuitBreaker,
    CircuitState,
    L2OrderBook,
    OrderBookAI,
    OrderBookConfig,
)


class TestL2OrderBook(unittest.TestCase):

    def setUp(self):
        self.book = L2OrderBook(symbol="btcusdt")

    def _make_snapshot(self, last_update_id: int = 100):
        return {
            "lastUpdateId": last_update_id,
            "bids": [["50000.0", "1.0"], ["49999.0", "2.0"], ["49998.0", "0.5"]],
            "asks": [["50001.0", "1.0"], ["50002.0", "2.0"], ["50003.0", "0.5"]],
        }

    # --- Snapshot ---
    def test_snapshot_applies_correctly(self):
        snap = self._make_snapshot(100)
        self.book.apply_snapshot(snap)

        self.assertTrue(self.book.is_synced)
        self.assertEqual(self.book.last_update_id, 100)
        self.assertAlmostEqual(self.book.bids[50000.0], 1.0)
        self.assertAlmostEqual(self.book.asks[50001.0], 1.0)

    # --- Delta: normal ---
    def test_valid_delta_updates_book(self):
        self.book.apply_snapshot(self._make_snapshot(100))
        delta = {
            "U": 101, "u": 102,
            "b": [["50000.0", "5.0"]],  # bid güncellemesi
            "a": [["50001.0", "0.0"]],  # ask silmesi
        }
        result = self.book.apply_delta(delta)

        self.assertTrue(result)
        self.assertEqual(self.book.last_update_id, 102)
        self.assertAlmostEqual(self.book.bids[50000.0], 5.0)
        self.assertNotIn(50001.0, self.book.asks)

    # --- Delta: sequence gap ---
    def test_sequence_gap_returns_false(self):
        self.book.apply_snapshot(self._make_snapshot(100))
        # U=105 > last_update_id+1=101 → gap
        delta = {"U": 105, "u": 110, "b": [], "a": []}
        result = self.book.apply_delta(delta)

        self.assertFalse(result)
        self.assertFalse(self.book.is_synced)

    # --- Delta: eski delta skip ---
    def test_old_delta_skipped_silently(self):
        self.book.apply_snapshot(self._make_snapshot(100))
        # u=99 < last_update_id+1=101 → eski, skip
        delta = {"U": 95, "u": 99, "b": [["50000.0", "999.0"]], "a": []}
        result = self.book.apply_delta(delta)

        self.assertTrue(result)
        self.assertAlmostEqual(self.book.bids[50000.0], 1.0)  # Değişmemeli
        self.assertEqual(self.book.last_update_id, 100)

    # --- Delta: sync olmadan ---
    def test_delta_without_sync_returns_false(self):
        delta = {"U": 1, "u": 2, "b": [], "a": []}
        result = self.book.apply_delta(delta)
        self.assertFalse(result)

    # --- Reset ---
    def test_reset_clears_state(self):
        self.book.apply_snapshot(self._make_snapshot(100))
        self.book.reset()

        self.assertFalse(self.book.is_synced)
        self.assertEqual(len(self.book.bids), 0)
        self.assertEqual(len(self.book.asks), 0)
        self.assertEqual(self.book.last_update_id, 0)

    # --- Top bid/ask ---
    def test_top_bid_ask(self):
        self.book.apply_snapshot(self._make_snapshot(100))
        self.assertAlmostEqual(self.book.top_bid(), 50000.0)
        self.assertAlmostEqual(self.book.top_ask(), 50001.0)

    def test_top_bid_ask_empty_returns_none(self):
        self.assertIsNone(self.book.top_bid())
        self.assertIsNone(self.book.top_ask())

    # --- Spread ---
    def test_spread_calculation(self):
        self.book.apply_snapshot(self._make_snapshot(100))
        spread = self.book.spread()
        self.assertAlmostEqual(spread, 1.0)

    # --- compute_metrics: price-sorted ---
    def test_compute_metrics_price_sorted(self):
        """
        Price-sorted olmayan bir dict sıralamasına güvenmemeli.
        Rastgele sırada eklenen bid/ask'ların en iyi N level'ı doğru seçmeli.
        """
        self.book.bids = {
            49990.0: 10.0,  # En iyi değil
            50000.0: 1.0,   # En iyi bid
            49995.0: 3.0,
        }
        self.book.asks = {
            50005.0: 10.0,  # Uzak
            50001.0: 1.0,   # En iyi ask
            50002.0: 2.0,
        }
        self.book.last_update_id = 1
        self.book.is_synced = True

        metrics = self.book.compute_metrics(levels=2)
        self.assertIsNotNone(metrics)

        # Top 2 bid: 50000 ve 49995 (49990 dışarıda kalmalı)
        # Top 2 ask: 50001 ve 50002 (50005 dışarıda kalmalı)
        self.assertAlmostEqual(metrics["top_bid"], 50000.0)
        self.assertAlmostEqual(metrics["top_ask"], 50001.0)

        # bid_vol = 1.0 + 3.0 = 4.0, ask_vol = 1.0 + 2.0 = 3.0
        self.assertAlmostEqual(metrics["bid_vol"], 4.0)
        self.assertAlmostEqual(metrics["ask_vol"], 3.0)
        expected_imbalance = 4.0 / 7.0
        self.assertAlmostEqual(metrics["imbalance"], expected_imbalance, places=4)

    # --- Imbalance boundary ---
    def test_imbalance_full_buy_pressure(self):
        self.book.bids = {50000.0: 100.0}
        self.book.asks = {50001.0: 0.001}  # Neredeyse sıfır satış
        self.book.is_synced = True
        metrics = self.book.compute_metrics(levels=5)
        self.assertGreater(metrics["imbalance"], 0.99)

    def test_imbalance_full_sell_pressure(self):
        self.book.bids = {50000.0: 0.001}
        self.book.asks = {50001.0: 100.0}
        self.book.is_synced = True
        metrics = self.book.compute_metrics(levels=5)
        self.assertLess(metrics["imbalance"], 0.01)

    # --- VWAP ---
    def test_vwap_near_mid_when_balanced(self):
        """Dengeli book'ta VWAP mid'e yakın olmalı."""
        self.book.bids = {50000.0: 10.0, 49999.0: 10.0}
        self.book.asks = {50001.0: 10.0, 50002.0: 10.0}
        self.book.is_synced = True
        metrics = self.book.compute_metrics(levels=10)
        mid = metrics["mid_price"]
        # %1'den az sapma beklenir
        self.assertAlmostEqual(metrics["vwap"], mid, delta=mid * 0.01)

    # --- compute_metrics: empty book ---
    def test_compute_metrics_empty_book(self):
        self.book.is_synced = True
        result = self.book.compute_metrics(levels=15)
        self.assertIsNone(result)

    # --- Spread average ---
    def test_spread_avg_not_available_with_few_samples(self):
        self.book.apply_snapshot(self._make_snapshot(100))
        self.book.spread()  # Tek sample
        self.assertIsNone(self.book.spread_avg())

    def test_spread_avg_available_after_10_samples(self):
        self.book.apply_snapshot(self._make_snapshot(100))
        for _ in range(10):
            self.book.spread()
        avg = self.book.spread_avg()
        self.assertIsNotNone(avg)
        self.assertAlmostEqual(avg, 1.0)


class TestCircuitBreaker(unittest.TestCase):

    def test_initial_state_closed(self):
        cb = CircuitBreaker("test")
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertTrue(cb.allow_request())

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)
        self.assertFalse(cb.allow_request())

    def test_transitions_to_half_open_after_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)
        time.sleep(0.02)
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)
        self.assertTrue(cb.allow_request())

    def test_success_resets_to_closed(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.record_success()
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertEqual(cb._failures, 0)

    def test_failure_count_resets_on_success(self):
        cb = CircuitBreaker("test", failure_threshold=5)
        for _ in range(3):
            cb.record_failure()
        cb.record_success()
        self.assertEqual(cb._failures, 0)


class TestOrderBookAIIntegration(unittest.IsolatedAsyncioTestCase):
    """
    Altyapı bağlantıları mock'lanarak analiz ve routing mantığı test edilir.
    """

    def _make_agent(self) -> OrderBookAI:
        config = OrderBookConfig(
            symbol="btcusdt",
            imbalance_threshold=0.65,
            depth_levels=15,
            redis_write_interval_ms=0,  # Throttle kapat
            spread_spike_multiplier=3.0,
            vwap_deviation_pct=0.5,
        )
        agent = OrderBookAI(config)
        agent.redis_client = AsyncMock()
        agent.redis_client.set = AsyncMock()
        agent.rabbitmq_channel = AsyncMock()
        agent.rabbitmq_channel.default_exchange = AsyncMock()
        agent.rabbitmq_channel.default_exchange.publish = AsyncMock()
        return agent

    def _set_balanced_book(self, agent: OrderBookAI):
        agent.book.bids = {50000.0: 5.0, 49999.0: 5.0}
        agent.book.asks = {50001.0: 5.0, 50002.0: 5.0}
        agent.book.is_synced = True
        agent.book.last_update_id = 100

    def _set_buy_pressure_book(self, agent: OrderBookAI):
        agent.book.bids = {50000.0: 100.0}
        agent.book.asks = {50001.0: 5.0}
        agent.book.is_synced = True

    def _set_sell_pressure_book(self, agent: OrderBookAI):
        agent.book.bids = {50000.0: 5.0}
        agent.book.asks = {50001.0: 100.0}
        agent.book.is_synced = True

    async def test_balanced_book_no_keda_trigger(self):
        agent = self._make_agent()
        self._set_balanced_book(agent)
        await agent._analyze_and_route()
        # Dengeli book'ta KEDA tetiklenmemeli
        agent.rabbitmq_channel.default_exchange.publish.assert_not_called()

    async def test_buy_pressure_triggers_keda(self):
        agent = self._make_agent()
        self._set_buy_pressure_book(agent)
        await agent._analyze_and_route()

        agent.rabbitmq_channel.default_exchange.publish.assert_called()
        call_args = agent.rabbitmq_channel.default_exchange.publish.call_args
        body = json.loads(call_args[0][0].body)
        self.assertEqual(body["event"], "HEAVY_BUY_PRESSURE")

    async def test_sell_pressure_triggers_keda(self):
        agent = self._make_agent()
        self._set_sell_pressure_book(agent)
        await agent._analyze_and_route()

        call_args = agent.rabbitmq_channel.default_exchange.publish.call_args
        body = json.loads(call_args[0][0].body)
        self.assertEqual(body["event"], "HEAVY_SELL_PRESSURE")

    async def test_redis_write_called_with_valid_metrics(self):
        agent = self._make_agent()
        self._set_balanced_book(agent)
        await agent._analyze_and_route()

        agent.redis_client.set.assert_called()
        call_args = agent.redis_client.set.call_args
        key = call_args[0][0]
        self.assertIn("btcusdt", key)
        payload = json.loads(call_args[0][1])
        self.assertIn("spread", payload)
        self.assertIn("vwap", payload)

    async def test_redis_throttle_prevents_excess_writes(self):
        config = OrderBookConfig(
            symbol="btcusdt",
            redis_write_interval_ms=1000,  # 1 saniye throttle
        )
        agent = OrderBookAI(config)
        agent.redis_client = AsyncMock()
        agent.redis_client.set = AsyncMock()
        agent.rabbitmq_channel = AsyncMock()
        agent.rabbitmq_channel.default_exchange = AsyncMock()
        agent.rabbitmq_channel.default_exchange.publish = AsyncMock()
        self._set_balanced_book(agent)

        # İki ardışık çağrı
        await agent._analyze_and_route()
        await agent._analyze_and_route()

        # Redis sadece bir kez yazılmalı
        self.assertEqual(agent.redis_client.set.call_count, 1)

    async def test_circuit_breaker_blocks_rabbitmq_on_failures(self):
        agent = self._make_agent()
        agent.rabbitmq_channel.default_exchange.publish = AsyncMock(
            side_effect=Exception("Connection refused")
        )
        self._set_buy_pressure_book(agent)

        # Circuit breaker threshold kadar hata üret
        for _ in range(agent.config.circuit_breaker_threshold):
            await agent._trigger_keda("TEST", {})

        # Circuit breaker açıldıktan sonra publish çağrılmamalı
        call_count_before = agent.rabbitmq_channel.default_exchange.publish.call_count
        await agent._trigger_keda("TEST", {})
        # Yeni çağrı olmamalı
        self.assertEqual(
            agent.rabbitmq_channel.default_exchange.publish.call_count,
            call_count_before,
        )

    async def test_analysis_does_not_raise_on_empty_book(self):
        agent = self._make_agent()
        agent.book.is_synced = True
        # Exception fırlatmamalı
        await agent._analyze_and_route()

    async def test_spread_spike_triggers_keda(self):
        agent = self._make_agent()
        agent.book.is_synced = True

        # Rolling average oluştur: 10 adet 1.0 spread
        agent.book.bids = {50000.0: 5.0}
        agent.book.asks = {50001.0: 5.0}
        for _ in range(10):
            agent.book.spread()

        # Şimdi büyük spread simüle et (3x threshold = 3.0, biz 5.0 veriyoruz)
        agent.book.bids = {50000.0: 5.0}
        agent.book.asks = {50005.0: 5.0}  # spread = 5.0 > avg(1.0) * 3.0

        await agent._analyze_and_route()

        publish_calls = agent.rabbitmq_channel.default_exchange.publish.call_args_list
        events = [json.loads(c[0][0].body)["event"] for c in publish_calls]
        self.assertIn("SPREAD_SPIKE", events)


if __name__ == "__main__":
    unittest.main(verbosity=2)