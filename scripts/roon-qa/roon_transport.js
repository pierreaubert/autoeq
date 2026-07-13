"use strict";

const RoonApi = require("node-roon-api");
const RoonApiTransport = require("node-roon-api-transport");

const action = process.argv[2] || "zones";
const qaZone = process.env.ROOMEQ_ROON_QA_ZONE;
const timeoutMs = Number(process.env.ROOMEQ_ROON_API_TIMEOUT_MS || 30000);
if (!qaZone) throw new Error("ROOMEQ_ROON_QA_ZONE is required");
if (!["zones", "play", "pause", "stop"].includes(action)) {
  throw new Error(`unsupported transport action: ${action}`);
}

let transport;
const zones = new Map();

function updateZones(command, data) {
  if (command === "Subscribed") {
    for (const zone of data.zones || []) zones.set(zone.zone_id, zone);
  }
  for (const zone of data.zones_added || []) zones.set(zone.zone_id, zone);
  for (const zone of data.zones_changed || []) zones.set(zone.zone_id, zone);
  for (const zone of data.zones_removed || []) zones.delete(zone.zone_id);

  const matches = [...zones.values()].filter((zone) => zone.display_name === qaZone);
  if (matches.length !== 1) return;
  const selectedZone = matches[0];
  if (action === "zones") {
    console.log(JSON.stringify({
      qa_zone: selectedZone.display_name,
      state: selectedZone.state,
      outputs: (selectedZone.outputs || []).map((output) => output.display_name),
    }));
    process.exit(0);
  }
  transport.control(selectedZone, action, (result) => {
    if (result !== "Success") throw new Error(`Roon transport ${action} failed: ${result}`);
    console.log(JSON.stringify({qa_zone: qaZone, action, result}));
    process.exit(0);
  });
}

const roon = new RoonApi({
  extension_id: "org.spinorama.sotf.roon-export-qa",
  display_name: "SotF Roon Export QA",
  display_version: "1.0.0",
  publisher: "SotF",
  email: "qa@invalid.example",
  website: "https://github.com/pierreaubert/autoeq",
  core_paired(core) {
    transport = core.services.RoonApiTransport;
    transport.subscribe_zones(updateZones);
  },
});

roon.init_services({required_services: [RoonApiTransport]});
roon.start_discovery();
setTimeout(() => {
  console.error(JSON.stringify({
    error: `exact QA zone '${qaZone}' was not discovered`,
    discovered_zone_names: [...zones.values()].map((zone) => zone.display_name),
    authorization: "Enable SotF Roon Export QA in Roon Settings > Extensions",
  }));
  process.exit(2);
}, timeoutMs);
