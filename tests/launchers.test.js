const test = require('node:test');
const assert = require('node:assert/strict');
const install = require('../install.js');
const update = require('../update.js');
const torch = require('../torch.js');

function matches(step, context) {
  if (!step.when) return true;
  const expression = step.when.slice(2, -2);
  return Function(...Object.keys(context), `return (${expression})`)(...Object.values(context));
}

function steps(script, context) {
  const executed = [];
  for (const step of script.run) {
    if (!matches(step, context)) continue;
    executed.push(step);
    if (step.next === null) break;
  }
  return executed;
}

test('updates surface pull failures and reuse installation for missing repos', () => {
  const actions = steps(update, { exists: () => false });
  assert.equal(actions.length, 2);
  assert.equal(actions[0].params.message, 'git pull --ff-only');
  assert.equal(actions[1].params.uri, 'install.js');
  assert.equal(actions[1].params.params.skip_start, true);
  const existing = steps(update, { exists: () => true });
  assert.deepEqual(existing.slice(1, 3).map(s => s.params.path), ['app/MOSS-TTS', 'app/MOSS-TTS-Nano']);
  assert(existing.slice(0, 3).every(s => s.params.message === 'git pull --ff-only'));
});

test('update install restores missing repos without starting a server', () => {
  const context = { platform: 'win32', arch: 'x64', gpu: 'nvidia', exists: () => false, args: { skip_start: true } };
  const actions = steps(install, context);
  assert.equal(actions.filter(s => String(s.params.message).startsWith('git clone')).length, 2);
  assert(!actions.some(s => s.params.uri === 'start.js'));
  assert(steps(install, { ...context, args: {} }).some(s => s.params.uri === 'start.js'));
});

test('platform branches do not fall through into a second torch install', () => {
  for (const [platform, arch, gpu] of [
    ['win32', 'x64', 'nvidia'], ['linux', 'x64', 'nvidia'],
    ['win32', 'x64', 'amd'], ['linux', 'x64', 'amd'],
    ['darwin', 'arm64', 'apple'], ['linux', 'x64', null],
  ]) {
    assert.equal(steps(torch, { platform, arch, gpu }).length, 1);
  }
  const unsupported = { platform: 'darwin', arch: 'x64', gpu: null };
  assert.equal(steps(install, unsupported)[0].method, 'notify');
  assert.equal(steps(install, unsupported).length, 1);
  assert.equal(steps(torch, unsupported)[0].method, 'notify');
});

test('reset remains visible after the environment is removed', async () => {
  const menu = await require('../pinokio.js').menu(null, {
    exists: () => false,
    running: script => script === 'reset.js',
  });
  assert.equal(menu[0].text, 'Resetting');
  assert.equal(menu[0].default, true);
});

test('server URL capture exposes the captured group', () => {
  const start = require('../start.js');
  const shell = start.run.find(s => s.method === 'shell.run');
  const pattern = shell.params.on[0].event;
  const match = new RegExp(pattern.slice(1, -1)).exec('Running on local URL:  http://127.0.0.1:8123');
  assert.equal(match[1], 'http://127.0.0.1:8123');
  assert.equal(start.run.find(s => s.method === 'local.set').params.url, '{{input.event[1]}}');
  assert.equal(start.daemon, true);
});
