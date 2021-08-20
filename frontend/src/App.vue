<template>
  <!-- <img alt="Vue logo" v-bind:src="src" :height="height" :width="width"/> -->
  <!-- <HelloWorld :msg="mess" /> -->
  <!-- <Card
    :alt="alt"
    :card="card"
    :suit="suit"
    :height="height"
    :width="width"
  /> -->
  <div id="app">
    <b-row class="text-center" align-v="center">
      <p>
        STATS:
        <br />
        TEAM CYBORG: {{ state.score_cyborg }}
        <br />
        TEAM AI: {{ state.score_ai }}
        <br />
        ROUND: {{ state.round }}
        <br />
        RUNG: {{ state.rung }}
      </p>
    </b-row>
    <b-container>
      <b-row alignContent="center" align-v="center" align-h="center">
        <b-col cols="8" align-v="center" alignContent="center">
          <CardStack
            position="top"
            :id="(state.player_id + 2) % 4"
            :key="(state.player_id + 2) % 4"
            :cards="state.player_cards[(state.player_id + 2) % 4].cards"
            :visible="state.player_cards[(state.player_id + 2) % 4].visible"
          />
        </b-col>
      </b-row>
      <b-row>
        <b-col lg="2" md="2">
          <CardStack
            position="left"
            :key="(state.player_id + 3) % 4"
            :id="(state.player_id + 3) % 4"
            :cards="state.player_cards[(state.player_id + 3) % 4].cards"
            :visible="state.player_cards[(state.player_id + 3) % 4].visible"
          />
        </b-col>
        <b-col lg="8" md="8"></b-col>
        <b-col lg="2" md="2">
          <CardStack
            position="right"
            :key="(state.player_id + 1) % 4"
            :id="(state.player_id + 1) % 4"
            :cards="state.player_cards[(state.player_id + 1) % 4].cards"
            :visible="state.player_cards[(state.player_id + 1) % 4].visible"
          />
        </b-col>
      </b-row>
      <b-row alignContent="center" align-v="center" align-h="center">
        <b-col cols="8" align-v="center" alignContent="center">
          <CardStack
            position="bottom"
            :key="state.player_id"
            :id="state.player_id"
            :cards="state.player_cards[state.player_id].cards"
            :visible="state.player_cards[state.player_id].visible"
          />
        </b-col>
      </b-row>
    </b-container>
  </div>
</template>

<script>
// import HelloWorld from "./components/HelloWorld.vue";
// import image from "./assets/public/static/images/cards/club/2.png";
// import Card from "./components/Card.vue";
import CardStack from "./components/CardStack.vue";

export default {
  name: "App",
  components: {
    // HelloWorld,
    // Card,
    CardStack,
  },
  data: function () {
    return {
      state: {
        player_id: 0,
        stacks: 4,
        score_cyborg: 5, // 0/2 will always be cyborg
        score_ai: 1, // 1/3 will always be ai
        round: 7,
        rung: "CLUBS",
        hand: [
          {
            player: 3,
            face: "ACE",
            suit: "SPADES",
          },
          {},
          {},
          {},
        ],
        hand_idx: 3,
        hand_started_by: 3,
        done: false,
        winner: null,
        current_player: 1,
        player_cards: [
          {
            cards: [
              {
                index: 0,
                face: "2",
                suit: "club",
                played: false,
                playable: true,
              },
            ],
            visible: true,
          },
          {
            cards: [
              {
                index: 1,
                face: "K",
                suit: "club",
                played: true,
                playable: true,
              },
            ],
            visible: false,
          },
          {
            cards: [
              {
                index: 2,
                face: "2",
                suit: "diamond",
                played: false,
                playable: true,
              },
            ],
            visible: false,
          },
          {
            cards: [
              {
                index: 3,
                face: "A",
                suit: "spade",
                played: false,
                playable: true,
              },
            ],
            visible: false,
          },
        ],
      },
    };
  },
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  /* text-align: center; */
  color: #2c3e50;
  margin-top: 60px;
}
</style>
