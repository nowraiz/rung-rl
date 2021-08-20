<template>
  <div>
    <!-- <center> -->
    <div v-for="(card, idx) in cards" :key="idx">
      <p>CARD: {{ card.face }} of {{ card.suit }}</p>
      <Card
        :alt="alt"
        :card="card.face"
        :suit="card.suit"
        :height="height"
        :width="width"
        :visible="visible"
      />
    </div>
    <!-- </center> -->
  </div>
</template>

<script>
import Card from "./Card.vue";

export default {
  components: { Card },
  props: {
    position: {
      type: String,
      required: true,
    },
    height: {
      type: String,
      default: "100",
      required: false,
    },
    width: {
      type: String,
      default: "100",
      required: false,
    },
    cards: {
      type: Array,
      default: () => {
        return [];
      },
    },
    visible: {
      type: Boolean,
      default: false,
    },
  },
  computed: {
    classObjectUpper: function () {
      if (this.position === "left" || this.position === "right") {
        return {
          "col-2": true,
        };
      } else {
        return {
          row: true,
        };
      }
    },
    classObject: function () {
      if (this.position === "left" || this.position === "right") {
        return {
          row: true,
          col: false,
        };
      } else {
        return {
          row: false,
          col: true,
        };
      }
    },
  },
};
</script>
<style scoped>
</style>