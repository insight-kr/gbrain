import { registerPostgresTests } from '../helpers/test-backends.ts';

await registerPostgresTests(
  () => import('../pathological/sinaxis-discovery-generations.test.ts'),
  () => import('../pathological/sinaxis-discovery-retirement-hold.test.ts'),
  () => import('../pathological/sinaxis-discovery-crash.test.ts'),
  () => import('../pathological/sinaxis-discovery-review.test.ts'),
);
